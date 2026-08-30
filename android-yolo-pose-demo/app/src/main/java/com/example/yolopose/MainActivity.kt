package com.example.yolopose

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.os.SystemClock
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.yolopose.databinding.ActivityMainBinding
import com.example.yolopose.ml.DetectionMode
import com.example.yolopose.ml.PersonTracker
import com.example.yolopose.ml.PoseResult
import com.example.yolopose.ml.TrackedPerson
import com.example.yolopose.ml.YoloPoseDetector
import com.example.yolopose.network.Esp32BleProvisioner
import com.example.yolopose.network.BackendConfig
import com.example.yolopose.network.Esp32MjpegFrameSource
import com.example.yolopose.network.Esp32ProvisioningState
import com.example.yolopose.network.Esp32ProvisioningStore
import com.example.yolopose.network.Esp32SnapshotFrameSource
import com.example.yolopose.network.Esp32WifiConnector
import com.example.yolopose.network.backend.BackendCoordinator
import com.example.yolopose.network.backend.UserProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private data class MovementSample(
        val timestampMs: Long,
        val area: Float
    )

    private enum class ScreenMode {
        HOME,
        AUTO_SELECT,
        COLOR_SELECT
    }

    private enum class HomeStage {
        PROVISION,
        LOGIN,
        REGISTER,
        DASHBOARD
    }

    private lateinit var binding: ActivityMainBinding
    private var detector: YoloPoseDetector? = null
    private val personTracker = PersonTracker()
    private lateinit var cameraExecutor: ExecutorService
    private var inferenceJob: Job? = null
    private var esp32WifiConnector: Esp32WifiConnector? = null
    private var esp32MjpegFrameSource: Esp32MjpegFrameSource? = null
    private var esp32FrameSource: Esp32SnapshotFrameSource? = null
    private var lastFrameTime = 0L
    private var lastInferenceStartedAt = 0L
    private var cameraStarted = false
    private var recognitionStarted = false
    private var useFrontCamera = false
    private var awaitingTemplateConfirmation = false
    private var confirmationDialogShowing = false
    private var latestSourceWidth = 0
    private var latestSourceHeight = 0
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private val movementSamples = ArrayDeque<MovementSample>()
    private var movementStatusText: String = ""
    private var screenMode: ScreenMode = ScreenMode.HOME
    private val backendCoordinator by lazy { BackendCoordinator(applicationContext) }
    private var backendBootstrapText: String = ""
    private var backendQuotaText: String = ""
    private var currentUserProfile: UserProfile? = null
    private var currentSessionId: String? = null
    private var sessionHeartbeatCounter = 0
    private var esp32PreviewScale = 45
    private var homeStage = HomeStage.PROVISION
    private val provisioningStore by lazy { Esp32ProvisioningStore(applicationContext) }
    private val bleProvisioner by lazy { Esp32BleProvisioner(applicationContext) }
    private var currentEsp32Host: String = DEFAULT_ESP32_HOST
    private var attemptingAutoProvision = false
    private var pendingConnectivityAction: (() -> Unit)? = null
    private var provisionReadyThisSession = false
    private var homeDeviceCheckRunning = false
    private var esp32SourceErrorCount = 0
    private var esp32RecoveryRunning = false
    private var esp32StatusProbeJob: Job? = null
    private var esp32StatusProbeFailureCount = 0

    private val startupPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            val locationGranted = result[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
            val wifiGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                result[Manifest.permission.NEARBY_WIFI_DEVICES] == true ||
                    ContextCompat.checkSelfPermission(this, Manifest.permission.NEARBY_WIFI_DEVICES) == PackageManager.PERMISSION_GRANTED
            } else {
                true
            }
            val bluetoothGranted = hasBluetoothProvisionPermissions()
            if (locationGranted && wifiGranted && bluetoothGranted) {
                pendingConnectivityAction?.invoke()
                pendingConnectivityAction = null
            } else {
                binding.homeSubtitleText.text = getString(R.string.esp32_permission_denied)
            }
        }
    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                binding.homeSubtitleText.text = getString(R.string.camera_permission_denied)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        val initialProvisioningConfig = provisioningStore.load()
        currentEsp32Host = initialProvisioningConfig.lastKnownHost ?: DEFAULT_ESP32_HOST
        bindProvisioningConfig(initialProvisioningConfig)
        provisionReadyThisSession = false
        setupActions()
        initDetector()
        showHomeMode()
        bootstrapBackend()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (screenMode == ScreenMode.HOME) {
                    when (homeStage) {
                        HomeStage.REGISTER -> showLoginPanel()
                        HomeStage.PROVISION -> {
                            isEnabled = false
                            onBackPressedDispatcher.onBackPressed()
                        }
                        HomeStage.DASHBOARD -> showProvisionStage()
                        HomeStage.LOGIN -> {
                            isEnabled = false
                            onBackPressedDispatcher.onBackPressed()
                        }
                    }
                } else {
                    showHomeMode()
                }
            }
        })
    }

    override fun onDestroy() {
        bleProvisioner.stop()
        esp32WifiConnector?.release()
        stopEsp32MjpegFrameSource()
        stopEsp32FrameSource()
        stopVideoRecording()
        lifecycleScope.launch(Dispatchers.IO) {
            backendCoordinator.endActiveSession(currentDisplayedFps(), buildSessionSummaryForBackend())
        }
        detector?.close()
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private fun initDetector() {
        if (!YoloPoseDetector.modelExists(this)) {
            binding.homeSubtitleText.text = getString(R.string.model_missing)
            return
        }
        runCatching { YoloPoseDetector(this) }
            .onSuccess { loadedDetector ->
                detector = loadedDetector
                refreshHomeSubtitle()
            }
            .onFailure { error ->
                binding.homeSubtitleText.text = getString(
                    R.string.model_loading_failed,
                    error.message ?: error.javaClass.simpleName
                )
            }
    }

    private fun bindProvisioningConfig(config: com.example.yolopose.network.Esp32ProvisioningConfig) {
        binding.hotspotSsidEditText.setText(config.hotspotSsid)
        binding.hotspotPasswordEditText.setText(config.hotspotPassword)
        binding.devicePrefixEditText.setText(config.deviceNamePrefix)
    }

    private fun saveProvisioningConfig() {
        val hotspotSsid = binding.hotspotSsidEditText.text?.toString()?.trim().orEmpty()
        val hotspotPassword = binding.hotspotPasswordEditText.text?.toString()?.trim().orEmpty()
        val devicePrefix = binding.devicePrefixEditText.text?.toString()?.trim().orEmpty()
        if (hotspotSsid.isBlank() || hotspotPassword.isBlank() || devicePrefix.isBlank()) {
            binding.homeSubtitleText.text = getString(R.string.provision_empty_fields)
            return
        }
        val newConfig = provisioningStore.load().copy(
            hotspotSsid = hotspotSsid,
            hotspotPassword = hotspotPassword,
            deviceNamePrefix = devicePrefix
        )
        runCatching {
            provisioningStore.save(newConfig)
        }.onSuccess {
            binding.homeSubtitleText.text = getString(R.string.provision_save_success)
            bleProvisioner.stop()
            attemptingAutoProvision = false
            provisionReadyThisSession = false
            homeDeviceCheckRunning = false
            showProvisionStage()
            requestConnectivityPermissionsThen {
                if (backendCoordinator.isAuthenticated()) {
                    prepareEsp32Connection(startSourceIfReady = false, allowBleFallback = false)
                } else {
                    showLoginPanel()
                    binding.homeSubtitleText.text = getString(R.string.provision_saved_wait_login_check)
                }
            }
        }.onFailure { error ->
            binding.homeSubtitleText.text = getString(
                R.string.provision_save_failed,
                error.message ?: error.javaClass.simpleName
            )
        }
    }

    private fun setupActions() {
        binding.startRecognitionButton.setOnClickListener {
            ensureAuthenticatedThen {
                enterRecognitionMode(ScreenMode.AUTO_SELECT)
            }
        }
        binding.selectColorButton.setOnClickListener {
            ensureAuthenticatedThen {
                enterRecognitionMode(ScreenMode.COLOR_SELECT)
            }
        }
        binding.loginButton.setOnClickListener {
            performLogin()
        }
        binding.registerButton.setOnClickListener {
            showRegisterStage()
        }
        binding.registerSubmitButton.setOnClickListener {
            performRegister()
        }
        binding.registerBackButton.setOnClickListener {
            showLoginPanel()
        }
        binding.saveProvisionButton.setOnClickListener {
            saveProvisioningConfig()
        }
        binding.logoutButton.setOnClickListener {
            backendCoordinator.logout()
            backendBootstrapText = getString(R.string.backend_not_configured, BackendConfig.baseUrl)
            backendQuotaText = ""
            binding.passwordEditText.setText("")
            binding.registerPasswordEditText.setText("")
            binding.registerConfirmPasswordEditText.setText("")
            updateAuthUi(false)
            refreshHomeSubtitle()
        }
        binding.recordButton.setOnClickListener {
            toggleRecording()
        }
        binding.previewSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                esp32PreviewScale = progress.coerceIn(0, 100)
                applyEsp32PreviewScale()
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
        binding.backButton.setOnClickListener {
            showHomeMode()
        }
        binding.poseOverlay.setOnTouchListener { _, event ->
            if (screenMode == ScreenMode.HOME) {
                return@setOnTouchListener false
            }
            if (event.action == MotionEvent.ACTION_UP) {
                handleTargetTap(event.x, event.y)
                true
            } else {
                true
            }
        }
    }

    private fun enterRecognitionMode(mode: ScreenMode) {
        if (!hasProvisioningConfig()) {
            showProvisionStage()
            binding.homeSubtitleText.text = getString(R.string.provision_stage_locked)
            return
        }
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        screenMode = mode
        recognitionStarted = true
        awaitingTemplateConfirmation = false
        confirmationDialogShowing = false
        movementSamples.clear()
        personTracker.reset()
        binding.poseOverlay.updateSelectionPoint(null, null)
        binding.poseOverlay.updatePose(null, 1, 1, DetectionMode.LOST, null, null, null, null, null, null, null, null, null, null)
        binding.modelStatusText.text = appendStatusSuffix(
            when (mode) {
            ScreenMode.AUTO_SELECT -> getString(R.string.tap_target_pick_hint)
            ScreenMode.COLOR_SELECT -> getString(R.string.tap_color_pick_hint)
            ScreenMode.HOME -> ""
            },
            getRecognitionNetworkStatus()
        )
        binding.resultText.text = when (mode) {
            ScreenMode.AUTO_SELECT -> getString(R.string.tap_target_detail)
            ScreenMode.COLOR_SELECT -> getString(R.string.tap_color_detail)
            ScreenMode.HOME -> ""
        }
        binding.fpsText.text = ""
        showRecognitionUi()
        openBackendSession(mode)
        requestConnectivityPermissionsThen {
            prepareEsp32Connection(startSourceIfReady = true, allowBleFallback = false)
        }
        ensureCameraPermission()
    }

    private fun bootstrapBackend() {
        lifecycleScope.launch(Dispatchers.IO) {
            val result = backendCoordinator.bootstrap()
            result.onSuccess {
                backendCoordinator.bindDefaultDevice()
            }
            val text = result.fold(
                onSuccess = { state ->
                    currentUserProfile = state.profile
                    backendQuotaText = getString(
                        R.string.backend_quota_template,
                        state.quotaSummary?.quotaRemain ?: state.profile?.quotaRemain ?: 0,
                        state.quotaSummary?.quotaTotal ?: state.profile?.quotaTotal ?: 0
                    )
                    getString(
                        R.string.backend_ready_template,
                        state.profile?.nickname ?: "",
                        BackendConfig.baseUrl
                    )
                },
                onFailure = { error ->
                    currentUserProfile = null
                    getString(R.string.backend_not_configured, BackendConfig.baseUrl)
                }
            )
            backendBootstrapText = text
            withContext(Dispatchers.Main) {
                binding.accountEditText.setText(backendCoordinator.savedAccount().orEmpty())
                updateAuthUi(backendCoordinator.isAuthenticated())
                showLoginOrProvisionStage()
                refreshHomeSubtitle()
            }
        }
    }

    private fun performLogin() {
        val account = binding.accountEditText.text?.toString()?.trim().orEmpty()
        val password = binding.passwordEditText.text?.toString().orEmpty()
        if (account.isBlank() || password.isBlank()) {
            binding.homeSubtitleText.text = getString(R.string.login_empty_fields)
            return
        }
        binding.homeSubtitleText.text = getString(R.string.backend_login_running)
        lifecycleScope.launch(Dispatchers.IO) {
            val result = backendCoordinator.login(account, password)
            result.onSuccess { state ->
                currentUserProfile = state.profile
                backendCoordinator.bindDefaultDevice()
                backendQuotaText = getString(
                    R.string.backend_quota_template,
                    state.quotaSummary?.quotaRemain ?: state.profile?.quotaRemain ?: 0,
                    state.quotaSummary?.quotaTotal ?: state.profile?.quotaTotal ?: 0
                )
                backendBootstrapText = getString(
                    R.string.backend_ready_template,
                    state.profile?.nickname ?: account,
                    BackendConfig.baseUrl
                )
                withContext(Dispatchers.Main) {
                    updateAuthUi(true)
                    showLoginOrProvisionStage()
                    refreshHomeSubtitle(extra = getString(R.string.backend_login_success))
                }
            }.onFailure { error ->
                currentUserProfile = null
                withContext(Dispatchers.Main) {
                    updateAuthUi(false)
                    binding.homeSubtitleText.text = getString(
                        R.string.backend_login_failed,
                        error.message ?: error.javaClass.simpleName
                    )
                }
            }
        }
    }

    private fun performRegister() {
        val nickname = binding.registerNicknameEditText.text?.toString()?.trim().orEmpty()
        val account = binding.registerAccountEditText.text?.toString()?.trim().orEmpty()
        val password = binding.registerPasswordEditText.text?.toString().orEmpty()
        val confirm = binding.registerConfirmPasswordEditText.text?.toString().orEmpty()
        if (nickname.isBlank() || account.isBlank() || password.isBlank() || confirm.isBlank()) {
            binding.homeSubtitleText.text = getString(R.string.login_empty_fields)
            return
        }
        if (password != confirm) {
            binding.homeSubtitleText.text = getString(R.string.register_password_mismatch)
            return
        }
        binding.homeSubtitleText.text = getString(R.string.backend_register_running)
        lifecycleScope.launch(Dispatchers.IO) {
            backendCoordinator.register(account, password, nickname)
                .onSuccess {
                    val loginResult = backendCoordinator.login(account, password)
                    loginResult.onSuccess { state ->
                        currentUserProfile = state.profile
                        backendCoordinator.bindDefaultDevice()
                        backendQuotaText = getString(
                            R.string.backend_quota_template,
                            state.quotaSummary?.quotaRemain ?: state.profile?.quotaRemain ?: 0,
                            state.quotaSummary?.quotaTotal ?: state.profile?.quotaTotal ?: 0
                        )
                        backendBootstrapText = getString(
                            R.string.backend_ready_template,
                            state.profile?.nickname ?: account,
                            BackendConfig.baseUrl
                        )
                        withContext(Dispatchers.Main) {
                            binding.accountEditText.setText(account)
                            binding.passwordEditText.setText(password)
                            clearRegisterForm()
                            updateAuthUi(true)
                            showLoginOrProvisionStage()
                            refreshHomeSubtitle(extra = getString(R.string.backend_register_and_login_success, account))
                            Toast.makeText(
                                this@MainActivity,
                                getString(R.string.backend_register_and_login_success, account),
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }.onFailure { error ->
                        withContext(Dispatchers.Main) {
                            binding.accountEditText.setText(account)
                            binding.passwordEditText.setText(password)
                            binding.homeSubtitleText.text = getString(
                                R.string.backend_login_failed,
                                error.message ?: error.javaClass.simpleName
                            )
                        }
                    }
                }
                .onFailure { error ->
                    withContext(Dispatchers.Main) {
                        binding.homeSubtitleText.text = getString(
                            R.string.backend_register_failed,
                            error.message ?: error.javaClass.simpleName
                        )
                    }
                }
        }
    }

    private fun ensureAuthenticatedThen(action: () -> Unit) {
        if (!backendCoordinator.isAuthenticated()) {
            binding.homeSubtitleText.text = getString(R.string.backend_login_required)
            return
        }
        lifecycleScope.launch(Dispatchers.IO) {
            val result = backendCoordinator.bootstrap()
            result.onSuccess { state ->
                currentUserProfile = state.profile
                backendQuotaText = getString(
                    R.string.backend_quota_template,
                    state.quotaSummary?.quotaRemain ?: state.profile?.quotaRemain ?: 0,
                    state.quotaSummary?.quotaTotal ?: state.profile?.quotaTotal ?: 0
                )
                backendBootstrapText = getString(
                    R.string.home_subtitle_ready,
                    state.profile?.nickname ?: "",
                    BackendConfig.baseUrl
                )
                withContext(Dispatchers.Main) {
                    updateAuthUi(true)
                    refreshHomeSubtitle()
                    action()
                }
            }.onFailure { error ->
                backendCoordinator.logout()
                currentUserProfile = null
                withContext(Dispatchers.Main) {
                    updateAuthUi(false)
                    binding.homeSubtitleText.text = getString(
                        R.string.backend_login_failed,
                        error.message ?: error.javaClass.simpleName
                    )
                }
            }
        }
    }

    private fun openBackendSession(mode: ScreenMode) {
        lifecycleScope.launch(Dispatchers.IO) {
            val result = backendCoordinator.startTrackingSession(
                mode = when (mode) {
                    ScreenMode.COLOR_SELECT -> "color_lock"
                    ScreenMode.AUTO_SELECT -> "person_lock"
                    ScreenMode.HOME -> "home"
                },
                videoSource = if (PREFER_STREAM_SOURCE) "esp32_stream" else "esp32_snapshot"
            )
            result.onSuccess { sessionId ->
                currentSessionId = sessionId
                sessionHeartbeatCounter = 0
            }.onFailure { error ->
                withContext(Dispatchers.Main) {
                    binding.resultText.text = appendStatusSuffix(
                        binding.resultText.text.toString(),
                        getString(R.string.backend_session_failed, error.message ?: error.javaClass.simpleName)
                    )
                }
            }
        }
    }

    private fun requestStartupPermissionsIfNeeded() {
        val permissions = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissions += Manifest.permission.ACCESS_FINE_LOCATION
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.NEARBY_WIFI_DEVICES) != PackageManager.PERMISSION_GRANTED
        ) {
            permissions += Manifest.permission.NEARBY_WIFI_DEVICES
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) != PackageManager.PERMISSION_GRANTED) {
                permissions += Manifest.permission.BLUETOOTH_SCAN
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
                permissions += Manifest.permission.BLUETOOTH_CONNECT
            }
        }
        if (permissions.isEmpty()) {
            pendingConnectivityAction?.invoke()
            pendingConnectivityAction = null
        } else {
            startupPermissionLauncher.launch(permissions.toTypedArray())
        }
    }

    private fun requestConnectivityPermissionsThen(action: () -> Unit) {
        pendingConnectivityAction = action
        runCatching {
            requestStartupPermissionsIfNeeded()
        }.onFailure { error ->
            pendingConnectivityAction = null
            binding.homeSubtitleText.text = getString(R.string.startup_safe_mode, error.message ?: error.javaClass.simpleName)
        }
    }

    private fun ensureCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun ensureEsp32WifiConnection() {
        if (isFinishing || isDestroyed) return
        currentEsp32Host = DEFAULT_ESP32_HOST
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            binding.homeSubtitleText.text = getString(R.string.esp32_connect_requires_q)
            if (screenMode != ScreenMode.HOME) {
                startEsp32InputSource()
            }
            return
        }
        if (esp32WifiConnector == null) {
            esp32WifiConnector = Esp32WifiConnector(
                context = this,
                ssid = ESP32_WIFI_SSID,
                password = ESP32_WIFI_PASSWORD,
                onConnected = { _ ->
                    runOnUiThread {
                        binding.homeSubtitleText.text = getString(R.string.esp32_connected_home, ESP32_WIFI_SSID)
                        if (screenMode != ScreenMode.HOME) {
                            binding.modelStatusText.text = appendStatusSuffix(
                                binding.modelStatusText.text.toString(),
                                appendStatusSuffix(
                                    getString(R.string.esp32_connected_status, ESP32_WIFI_SSID),
                                    getRecognitionNetworkStatus()
                                )
                            )
                            startEsp32InputSource()
                        }
                    }
                },
                onUnavailable = { message ->
                    runOnUiThread {
                        if (screenMode == ScreenMode.HOME) {
                            binding.homeSubtitleText.text = getString(R.string.esp32_connecting_home)
                        } else {
                            binding.modelStatusText.text = getString(R.string.esp32_error_status, message)
                        }
                    }
                }
            )
        }
        if (esp32WifiConnector?.isAlreadyConnected() == true) {
            if (screenMode != ScreenMode.HOME) {
                startEsp32InputSource()
            }
            binding.homeSubtitleText.text = getString(R.string.esp32_connected_home, ESP32_WIFI_SSID)
            return
        }
        if (screenMode == ScreenMode.HOME) {
            binding.homeSubtitleText.text = getString(R.string.esp32_connecting_home)
        } else {
            binding.modelStatusText.text = appendStatusSuffix(
                getString(R.string.esp32_connecting_status, ESP32_WIFI_SSID),
                getRecognitionNetworkStatus()
            )
        }
        runCatching {
            esp32WifiConnector?.connectIfNeeded()
        }.onFailure { error ->
            if (screenMode == ScreenMode.HOME) {
                binding.homeSubtitleText.text = getString(R.string.esp32_error_status, error.message ?: error.javaClass.simpleName)
            } else {
                binding.modelStatusText.text = getString(R.string.esp32_error_status, error.message ?: error.javaClass.simpleName)
            }
        }
    }

    private fun tryAutomaticEsp32Provisioning(startSourceIfReady: Boolean) {
        if (attemptingAutoProvision) return
        if (!backendCoordinator.isAuthenticated()) {
            binding.homeSubtitleText.text = getString(R.string.provision_saved_wait_login_check)
            return
        }
        val config = provisioningStore.load()
        currentEsp32Host = config.lastKnownHost ?: DEFAULT_ESP32_HOST
        if (!config.autoProvisionEnabled || config.hotspotSsid.isBlank() || config.hotspotPassword.isBlank()) {
            binding.homeSubtitleText.text = getString(R.string.provision_stage_locked)
            return
        }
        if (!hasBluetoothProvisionPermissions()) {
            binding.homeSubtitleText.text = getString(R.string.esp32_auto_provision_permission_missing)
            return
        }
        attemptingAutoProvision = true
        updateProvisioningStatus(Esp32ProvisioningState.Scanning)
        bleProvisioner.startProvisioning(config) { state ->
            handleProvisioningState(state, startSourceIfReady)
        }
    }

    private fun prepareEsp32Connection(
        startSourceIfReady: Boolean,
        allowBleFallback: Boolean = true
    ) {
        updateEsp32DiscoveryStatus(
            getString(R.string.esp32_discovery_wait_ip)
        )
        lifecycleScope.launch(Dispatchers.IO) {
            val backendHost = fetchBackendLatestEsp32Host()
            // 后端已经拿到本次 IP 时，ESP32 可能还处在 STOP_BLE 到 CAMERA_ON 的过渡阶段，
            // 这里先等设备把相机 HTTP 服务真正启动，再判定是否可访问。
            val resolvedHost = backendHost?.let { waitForEsp32ServiceStartup(it) }
            withContext(Dispatchers.Main) {
                if (resolvedHost != null) {
                    markProvisionReady(resolvedHost)
                    if (startSourceIfReady) {
                        startEsp32InputSource()
                    }
                } else {
                    if (allowBleFallback) {
                        tryAutomaticEsp32Provisioning(startSourceIfReady)
                    } else {
                        handleDirectConnectionUnavailable()
                    }
                }
            }
        }
    }

    private fun handleDirectConnectionUnavailable() {
        if (screenMode == ScreenMode.HOME) {
            showProvisionStage()
            binding.homeSubtitleText.text = getString(R.string.esp32_direct_unavailable_need_manual)
        } else {
            binding.modelStatusText.text = appendStatusSuffix(
                getString(R.string.esp32_direct_unavailable_need_manual),
                getRecognitionNetworkStatus()
            )
        }
    }

    private fun handleProvisioningState(
        state: Esp32ProvisioningState,
        startSourceIfReady: Boolean
    ) {
        updateProvisioningStatus(state)
        when (state) {
            is Esp32ProvisioningState.Completed -> {
                attemptingAutoProvision = false
                markProvisionReady(state.esp32Host)
                if (startSourceIfReady) {
                    startEsp32InputSource()
                }
            }
            is Esp32ProvisioningState.Failed -> {
                attemptingAutoProvision = false
                recoverFromProvisioningFailureIfHostReachable(startSourceIfReady)
            }
            else -> Unit
        }
    }

    private fun recoverFromProvisioningFailureIfHostReachable(startSourceIfReady: Boolean) {
        lifecycleScope.launch(Dispatchers.IO) {
            val backendHost = fetchBackendLatestEsp32Host()
            val resolvedHost = backendHost?.let { waitForEsp32ServiceStartup(it) }
            withContext(Dispatchers.Main) {
                if (resolvedHost == null) return@withContext
                markProvisionReady(resolvedHost)
                if (startSourceIfReady) {
                    startEsp32InputSource()
                }
            }
        }
    }

    private fun markProvisionReady(host: String) {
        currentEsp32Host = host
        provisioningStore.updateLastKnownHost(host)
        provisionReadyThisSession = true
        homeDeviceCheckRunning = false
        if (screenMode == ScreenMode.HOME) {
            showLoginOrProvisionStage()
            if (backendCoordinator.isAuthenticated()) {
                binding.homeSubtitleText.text = getString(R.string.provision_stage_ready)
            } else {
                binding.homeSubtitleText.text = getString(R.string.esp32_auto_provision_ready, host)
            }
        } else {
            binding.modelStatusText.text = appendStatusSuffix(
                getString(R.string.esp32_session_ip, host),
                getRecognitionNetworkStatus()
            )
        }
    }

    private fun updateProvisioningStatus(state: Esp32ProvisioningState) {
        val text = when (state) {
            Esp32ProvisioningState.Scanning -> getString(R.string.esp32_auto_provision_scanning)
            is Esp32ProvisioningState.DeviceFound -> getString(R.string.esp32_auto_provision_found, state.deviceName)
            is Esp32ProvisioningState.DeviceConnected -> getString(R.string.esp32_auto_provision_connected, state.deviceName)
            Esp32ProvisioningState.DiscoveringServices -> getString(R.string.esp32_auto_provision_discovering)
            Esp32ProvisioningState.ServicesReady -> getString(R.string.esp32_auto_provision_services_ready)
            is Esp32ProvisioningState.SendingHotspotConfig -> getString(R.string.esp32_auto_provision_sending, state.ssid)
            is Esp32ProvisioningState.WaitingForDeviceJoin -> getString(R.string.esp32_auto_provision_waiting, state.ssid)
            is Esp32ProvisioningState.Completed -> getString(R.string.esp32_auto_provision_ready, state.esp32Host)
            is Esp32ProvisioningState.Failed -> getString(R.string.esp32_auto_provision_failed, state.message)
        }
        if (screenMode == ScreenMode.HOME) {
            binding.homeSubtitleText.text = text
        } else {
            binding.modelStatusText.text = appendStatusSuffix(text, getRecognitionNetworkStatus())
        }
    }

    private fun hasBluetoothProvisionPermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun startCamera() {
        if (screenMode == ScreenMode.HOME) return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            runCatching {
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().apply {
                    setSurfaceProvider(binding.previewView.surfaceProvider)
                }
                val recorder = Recorder.Builder()
                    .setExecutor(cameraExecutor)
                    .setQualitySelector(QualitySelector.from(Quality.SD))
                    .build()
                videoCapture = VideoCapture.withOutput(recorder)

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    buildCameraSelector(),
                    preview,
                    videoCapture
                )
                cameraStarted = true
                updateRecordingUi()
            }.onFailure { error ->
                binding.modelStatusText.text = getString(R.string.camera_start_failed, error.message ?: error.javaClass.simpleName)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(bitmap: android.graphics.Bitmap) {
        if (!recognitionStarted || screenMode == ScreenMode.HOME) {
            return
        }
        val now = SystemClock.elapsedRealtime()
        if (now - lastInferenceStartedAt < INFERENCE_INTERVAL_MS) {
            return
        }
        latestSourceWidth = bitmap.width
        latestSourceHeight = bitmap.height
        if (inferenceJob?.isActive == true) {
            return
        }
        lastInferenceStartedAt = now
        inferenceJob = lifecycleScope.launch {
            val detectorNeeded = !(screenMode == ScreenMode.COLOR_SELECT && personTracker.shouldSkipPoseInference())
            val result = if (detectorNeeded) {
                detector?.estimatePose(bitmap)
            } else {
                null
            }
            val frameTime = SystemClock.elapsedRealtime()
            val tracked = withContext(Dispatchers.Default) {
                personTracker.update(result, bitmap, frameTime)
            }
            renderEsp32Frame(bitmap)
            val elapsedSeconds = personTracker.initializationElapsedSeconds(frameTime)
            if (personTracker.isColorPickPending()) {
                binding.modelStatusText.text = getString(R.string.mode_color_pick_pending)
            } else if (personTracker.isInitializing()) {
                binding.modelStatusText.text = getString(
                    R.string.init_status_template,
                    elapsedSeconds.coerceAtMost(INIT_SECONDS),
                    INIT_SECONDS
                )
                if (elapsedSeconds >= INIT_SECONDS) {
                    val ready = personTracker.finishInitialization()
                    binding.modelStatusText.text = if (ready) {
                        awaitingTemplateConfirmation = true
                        maybeShowTemplateConfirmationDialog()
                        getString(R.string.init_ready_wait_confirm)
                    } else {
                        personTracker.startInitialization(frameTime)
                        getString(R.string.init_need_target)
                    }
                }
            } else {
                binding.modelStatusText.text = if (awaitingTemplateConfirmation) {
                    getString(R.string.init_ready_wait_confirm)
                } else if (personTracker.isTemplateConfirmed()) {
                    if (tracked.skinColor == null) {
                        getString(R.string.template_color_locked_status)
                    } else {
                        getString(R.string.mode_color_locked)
                    }
                } else {
                    getString(R.string.model_ready_simple)
                }
            }
            updateMovementState(tracked, frameTime)
            maybeSyncBackend(tracked)
            binding.poseOverlay.updatePose(
                tracked.poseResult,
                bitmap.width,
                bitmap.height,
                tracked.mode,
                tracked.averageColor,
                tracked.anchorColor,
                tracked.trackingBox,
                tracked.skinBox,
                tracked.swimwearBox,
                tracked.skinContour,
                tracked.swimwearContour,
                tracked.skinCells,
                tracked.swimwearCells,
                tracked.colorComboBox
            )
            binding.resultText.text = buildTrackedStatus(tracked)
            updateFps()
        }
    }

    private fun buildTrackedStatus(tracked: TrackedPerson): String {
        if (tracked.mode == DetectionMode.LOST) {
            return getString(R.string.track_lost)
        }
        val result = tracked.poseResult
        val header = buildString {
            append(getString(R.string.track_id_template, tracked.trackId))
            append(" | ")
            append(
                getString(
                    when (tracked.mode) {
                        DetectionMode.INITIALIZING -> R.string.mode_initializing
                        DetectionMode.COLOR_TRACK -> R.string.mode_color_tracking
                        DetectionMode.FULL_BODY -> R.string.mode_full_body
                        DetectionMode.LEG_ONLY -> R.string.mode_leg_only
                        DetectionMode.FOOT_ONLY -> R.string.mode_foot_only
                        DetectionMode.TRACKED_MEMORY -> R.string.mode_tracked
                        DetectionMode.LOST -> R.string.track_lost
                    }
                )
            )
        }
        val colorText = tracked.averageColor?.let(::formatColorText) ?: getString(R.string.track_color_unknown)
        val anchorText = tracked.anchorColor?.let(::formatAnchorColorText) ?: getString(R.string.anchor_color_unknown)
        val lockText = if (personTracker.isTemplateConfirmed()) {
            if (tracked.skinColor == null) {
                getString(R.string.template_color_locked_status)
            } else {
                getString(R.string.template_locked_status)
            }
        } else if (awaitingTemplateConfirmation) {
            getString(R.string.template_wait_confirm_status)
        } else {
            ""
        }
        val bodyText = if (result != null) {
            getString(
                R.string.image_test_result_detected,
                tracked.confidence,
                result.keyPoints.count { it.confidence > 0.3f }
            )
        } else if (tracked.mode == DetectionMode.COLOR_TRACK || tracked.mode == DetectionMode.TRACKED_MEMORY) {
            getString(R.string.color_only_tracking_status, tracked.confidence)
        } else {
            getString(R.string.image_test_result_failed)
        }
        val legText = result?.let(::buildLegStatus).orEmpty()
        val sourceText = getString(
            R.string.esp32_source_status,
            if (PREFER_STREAM_SOURCE) {
                getString(R.string.esp32_source_stream, buildEsp32StreamUrl())
            } else {
                getString(R.string.esp32_source_snapshot, buildEsp32CaptureUrl())
            }
        )
        val recordText = if (activeRecording != null) {
            getString(R.string.phone_recording_running)
        } else {
            getString(R.string.phone_recording_waiting)
        }
        val movementText = currentMovementStatus()
        return listOf(header, lockText, sourceText, recordText, movementText, bodyText, colorText, anchorText, legText)
            .filter { it.isNotBlank() }
            .joinToString("\n")
    }

    private fun formatColorText(color: Int): String {
        val hex = String.format(
            "#%02X%02X%02X",
            android.graphics.Color.red(color),
            android.graphics.Color.green(color),
            android.graphics.Color.blue(color)
        )
        return getString(R.string.track_color_template, hex)
    }

    private fun formatAnchorColorText(color: Int): String {
        val hex = String.format(
            "#%02X%02X%02X",
            android.graphics.Color.red(color),
            android.graphics.Color.green(color),
            android.graphics.Color.blue(color)
        )
        return getString(R.string.anchor_color_template, hex)
    }

    private fun buildLegStatus(result: PoseResult): String {
        val leftOk = isLegDetected(result, hip = 11, knee = 13, ankle = 15)
        val rightOk = isLegDetected(result, hip = 12, knee = 14, ankle = 16)
        val leftFoot = result.keyPoints.getOrNull(15)?.confidence?.let { it > 0.3f } == true
        val rightFoot = result.keyPoints.getOrNull(16)?.confidence?.let { it > 0.3f } == true
        return getString(
            R.string.leg_foot_result_detected,
            if (leftOk) getString(R.string.leg_status_ok) else getString(R.string.leg_status_missing),
            if (rightOk) getString(R.string.leg_status_ok) else getString(R.string.leg_status_missing),
            if (leftFoot) getString(R.string.leg_status_ok) else getString(R.string.leg_status_missing),
            if (rightFoot) getString(R.string.leg_status_ok) else getString(R.string.leg_status_missing)
        )
    }

    private fun isLegDetected(result: PoseResult, hip: Int, knee: Int, ankle: Int): Boolean {
        return listOf(hip, knee, ankle).all { index ->
            result.keyPoints.getOrNull(index)?.confidence?.let { it > 0.3f } == true
        }
    }

    private fun updateFps() {
        val now = SystemClock.elapsedRealtime()
        if (lastFrameTime != 0L) {
            val fps = 1000f / (now - lastFrameTime).coerceAtLeast(1L)
            binding.fpsText.text = getString(R.string.fps_template, fps)
        }
        lastFrameTime = now
    }

    private fun buildCameraSelector(): CameraSelector {
        return CameraSelector.Builder()
            .requireLensFacing(
                if (useFrontCamera) CameraSelector.LENS_FACING_FRONT
                else CameraSelector.LENS_FACING_BACK
            )
            .build()
    }

    private fun maybeShowTemplateConfirmationDialog() {
        if (!awaitingTemplateConfirmation || confirmationDialogShowing || !personTracker.templateReady()) {
            return
        }
        confirmationDialogShowing = true
        AlertDialog.Builder(this)
            .setTitle(R.string.template_confirm_title)
            .setMessage(R.string.template_confirm_message)
            .setCancelable(false)
            .setPositiveButton(R.string.template_confirm_yes) { dialog, _ ->
                personTracker.confirmTemplateLock()
                awaitingTemplateConfirmation = false
                confirmationDialogShowing = false
                binding.modelStatusText.text = getString(R.string.mode_color_locked)
                dialog.dismiss()
            }
            .setNegativeButton(R.string.template_confirm_retry) { dialog, _ ->
                awaitingTemplateConfirmation = false
                confirmationDialogShowing = false
                personTracker.startInitialization(SystemClock.elapsedRealtime())
                binding.modelStatusText.text = getString(R.string.init_need_target)
                dialog.dismiss()
            }
            .setOnDismissListener {
                confirmationDialogShowing = false
            }
            .show()
    }

    private fun handleTargetTap(viewX: Float, viewY: Float) {
        if (latestSourceWidth <= 0 || latestSourceHeight <= 0) {
            return
        }
        val mapped = mapViewPointToSource(viewX, viewY) ?: return
        binding.poseOverlay.updateSelectionPoint(mapped.first, mapped.second)
        recognitionStarted = true
        awaitingTemplateConfirmation = false
        confirmationDialogShowing = false
        if (screenMode == ScreenMode.COLOR_SELECT) {
            personTracker.startDirectColorLock(mapped.first, mapped.second, SystemClock.elapsedRealtime())
            binding.modelStatusText.text = getString(R.string.tap_color_status)
            binding.resultText.text = getString(R.string.tap_color_detail)
        } else {
            personTracker.setSelectionPoint(mapped.first, mapped.second)
            personTracker.startInitialization(SystemClock.elapsedRealtime())
            binding.modelStatusText.text = getString(R.string.tap_target_status)
            binding.resultText.text = getString(R.string.tap_target_detail)
        }
    }

    private fun mapViewPointToSource(viewX: Float, viewY: Float): Pair<Float, Float>? {
        val sourceWidth = latestSourceWidth.toFloat()
        val sourceHeight = latestSourceHeight.toFloat()
        if (sourceWidth <= 0f || sourceHeight <= 0f) return null
        val overlayWidth = binding.poseOverlay.width.toFloat()
        val overlayHeight = binding.poseOverlay.height.toFloat()
        if (overlayWidth <= 0f || overlayHeight <= 0f) return null

        val scale = minOf(overlayWidth / sourceWidth, overlayHeight / sourceHeight)
        val drawnWidth = sourceWidth * scale
        val drawnHeight = sourceHeight * scale
        val offsetX = (overlayWidth - drawnWidth) / 2f
        val offsetY = (overlayHeight - drawnHeight) / 2f
        if (viewX !in offsetX..(offsetX + drawnWidth) || viewY !in offsetY..(offsetY + drawnHeight)) {
            return null
        }
        val sourceX = ((viewX - offsetX) / scale).coerceIn(0f, sourceWidth)
        val sourceY = ((viewY - offsetY) / scale).coerceIn(0f, sourceHeight)
        return Pair(sourceX, sourceY)
    }

    private fun showHomeMode() {
        // 返回首页时主动停止残留 BLE 配网，避免旧任务超时后把错误文案刷回首页。
        bleProvisioner.stop()
        attemptingAutoProvision = false
        inferenceJob?.cancel()
        lifecycleScope.launch(Dispatchers.IO) {
            backendCoordinator.endActiveSession(currentDisplayedFps(), buildSessionSummaryForBackend())
        }
        recognitionStarted = false
        awaitingTemplateConfirmation = false
        confirmationDialogShowing = false
        movementSamples.clear()
        movementStatusText = ""
        screenMode = ScreenMode.HOME
        currentSessionId = null
        sessionHeartbeatCounter = 0
        personTracker.reset()
        stopEsp32MjpegFrameSource()
        stopEsp32FrameSource()
        stopEsp32StatusProbe()
        stopVideoRecording()
        binding.previewView.visibility = View.GONE
        binding.topHintCard.visibility = View.GONE
        binding.esp32PreviewCard.visibility = View.GONE
        binding.testImageView.setImageDrawable(null)
        binding.statusCard.visibility = View.GONE
        binding.controlBar.visibility = View.GONE
        binding.backButton.visibility = View.GONE
        binding.homeMenu.visibility = View.VISIBLE
        binding.recordButton.text = getString(R.string.record_start)
        binding.poseOverlay.updateSelectionPoint(null, null)
        binding.poseOverlay.updatePose(null, 1, 1, DetectionMode.LOST, null, null, null, null, null, null, null, null, null, null)
        binding.resultText.text = ""
        binding.fpsText.text = ""
        showLoginOrProvisionStage()
        refreshHomeSubtitle()
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        if (cameraStarted) {
            ProcessCameraProvider.getInstance(this).get().unbindAll()
            cameraStarted = false
        }
        videoCapture = null
    }

    private fun showRecognitionUi() {
        binding.homeMenu.visibility = View.GONE
        binding.previewView.visibility = View.VISIBLE
        binding.topHintCard.visibility = View.VISIBLE
        binding.esp32PreviewCard.visibility = View.VISIBLE
        binding.statusCard.visibility = View.VISIBLE
        binding.controlBar.visibility = View.VISIBLE
        binding.backButton.visibility = View.VISIBLE
        binding.previewSizeSeekBar.progress = esp32PreviewScale
        updateRecordingUi()
        binding.root.post {
            applyEsp32PreviewScale()
        }
    }

    private fun startEsp32FrameSource() {
        val connector = esp32WifiConnector
        if (currentEsp32Host == DEFAULT_ESP32_HOST && connector?.activeNetwork == null && connector?.isAlreadyConnected() != true) {
            return
        }
        stopEsp32MjpegFrameSource()
        stopEsp32FrameSource()
        esp32FrameSource = Esp32SnapshotFrameSource(
            scope = lifecycleScope,
            snapshotUrl = buildEsp32CaptureUrl(),
            networkProvider = { if (currentEsp32Host == DEFAULT_ESP32_HOST) esp32WifiConnector?.activeNetwork else null },
            pollIntervalMs = ESP32_POLL_INTERVAL_MS,
            onFrame = { bitmap ->
                esp32SourceErrorCount = 0
                analyzeFrame(bitmap)
            },
            onError = { message ->
                if (screenMode != ScreenMode.HOME) {
                    binding.modelStatusText.text = getString(R.string.esp32_error_status, message)
                }
                handleEsp32SourceFailure(message)
            }
        ).also { it.start() }
    }

    private fun startEsp32MjpegFrameSource() {
        val connector = esp32WifiConnector
        if (currentEsp32Host == DEFAULT_ESP32_HOST && connector?.activeNetwork == null && connector?.isAlreadyConnected() != true) {
            return
        }
        stopEsp32FrameSource()
        stopEsp32MjpegFrameSource()
        esp32MjpegFrameSource = Esp32MjpegFrameSource(
            scope = lifecycleScope,
            streamUrl = buildEsp32StreamUrl(),
            networkProvider = { if (currentEsp32Host == DEFAULT_ESP32_HOST) esp32WifiConnector?.activeNetwork else null },
            onFrame = { bitmap ->
                esp32SourceErrorCount = 0
                analyzeFrame(bitmap)
            },
            onError = { message ->
                if (screenMode != ScreenMode.HOME) {
                    binding.modelStatusText.text = getString(R.string.esp32_stream_error_status, message)
                }
                handleEsp32SourceFailure(message, fallbackToSnapshot = true)
            }
        ).also { it.start() }
    }

    private fun startEsp32InputSource() {
        startEsp32StatusProbe()
        if (PREFER_STREAM_SOURCE) {
            startEsp32MjpegFrameSource()
        } else {
            startEsp32FrameSource()
        }
    }

    private fun stopEsp32MjpegFrameSource() {
        esp32MjpegFrameSource?.stop()
        esp32MjpegFrameSource = null
    }

    private fun stopEsp32FrameSource() {
        esp32FrameSource?.stop()
        esp32FrameSource = null
    }

    private fun startEsp32StatusProbe() {
        if (screenMode == ScreenMode.HOME) return
        val host = currentEsp32Host.takeIf { it.isNotBlank() } ?: return
        if (host == DEFAULT_ESP32_HOST) return
        esp32StatusProbeJob?.cancel()
        esp32StatusProbeFailureCount = 0
        esp32StatusProbeJob = lifecycleScope.launch(Dispatchers.IO) {
            while (screenMode != ScreenMode.HOME) {
                val reachable = isEsp32StatusReachable(host)
                withContext(Dispatchers.Main) {
                    if (screenMode == ScreenMode.HOME) return@withContext
                    if (reachable) {
                        esp32StatusProbeFailureCount = 0
                    } else {
                        esp32StatusProbeFailureCount += 1
                        if (esp32StatusProbeFailureCount >= ESP32_STATUS_PROBE_FAILURE_THRESHOLD) {
                            handleEsp32SourceFailure("status probe failed")
                        }
                    }
                }
                delay(ESP32_STATUS_PROBE_INTERVAL_MS)
            }
        }
    }

    private fun stopEsp32StatusProbe() {
        esp32StatusProbeJob?.cancel()
        esp32StatusProbeJob = null
        esp32StatusProbeFailureCount = 0
    }

    private fun handleEsp32SourceFailure(
        message: String,
        fallbackToSnapshot: Boolean = false
    ) {
        if (screenMode == ScreenMode.HOME) return
        esp32SourceErrorCount += 1
        if (fallbackToSnapshot && esp32SourceErrorCount < ESP32_SOURCE_FAILURE_THRESHOLD) {
            startEsp32FrameSource()
            return
        }
        if (esp32SourceErrorCount < ESP32_SOURCE_FAILURE_THRESHOLD || esp32RecoveryRunning) {
            return
        }
        esp32RecoveryRunning = true
        lifecycleScope.launch(Dispatchers.IO) {
            val backendHost = fetchBackendLatestEsp32Host()
            val resolvedHost = backendHost?.let { resolveReachableEsp32HostWithRetries(it) }
            withContext(Dispatchers.Main) {
                esp32RecoveryRunning = false
                if (screenMode == ScreenMode.HOME) return@withContext
                if (resolvedHost != null) {
                    esp32SourceErrorCount = 0
                    currentEsp32Host = resolvedHost
                    provisioningStore.updateLastKnownHost(resolvedHost)
                    binding.modelStatusText.text = appendStatusSuffix(
                        getString(R.string.esp32_stream_recovering, resolvedHost),
                        getRecognitionNetworkStatus()
                    )
                    startEsp32InputSource()
                    return@withContext
                }
                enterProvisionRecoveryMode(message)
            }
        }
    }

    private fun enterProvisionRecoveryMode(reason: String) {
        esp32SourceErrorCount = 0
        provisionReadyThisSession = false
        currentEsp32Host = DEFAULT_ESP32_HOST
        showHomeMode()
        showProvisionStage()
        binding.homeSubtitleText.text = getString(R.string.esp32_runtime_reprovision_needed, reason)
        requestConnectivityPermissionsThen {
            tryAutomaticEsp32Provisioning(startSourceIfReady = false)
        }
    }

    private fun renderEsp32Frame(bitmap: android.graphics.Bitmap) {
        binding.testImageView.setImageBitmap(bitmap)
    }

    private fun startVideoRecording() {
        if (activeRecording != null || screenMode == ScreenMode.HOME) return
        val capture = videoCapture ?: return
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "swimming_phone_record_$timestamp")
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/SwimmingTool")
        }
        val mediaStoreOutput = androidx.camera.video.MediaStoreOutputOptions.Builder(
            contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        ).setContentValues(contentValues).build()

        runCatching {
            activeRecording = capture.output
                .prepareRecording(this, mediaStoreOutput)
                .start(ContextCompat.getMainExecutor(this)) { event ->
                    when (event) {
                        is VideoRecordEvent.Start -> {
                            updateRecordingUi()
                            if (screenMode != ScreenMode.HOME) {
                                binding.modelStatusText.text = appendStatusSuffix(
                                    binding.modelStatusText.text.toString(),
                                    getString(R.string.phone_recording_running_short)
                                )
                            }
                        }
                        is VideoRecordEvent.Finalize -> {
                            activeRecording = null
                            updateRecordingUi()
                            if (screenMode != ScreenMode.HOME && !event.hasError()) {
                                binding.modelStatusText.text = appendStatusSuffix(
                                    binding.modelStatusText.text.toString(),
                                    getString(R.string.phone_record_saved)
                                )
                            }
                        }
                    }
                }
        }.onFailure { error ->
            updateRecordingUi()
            binding.modelStatusText.text = getString(R.string.phone_recording_failed, error.message ?: error.javaClass.simpleName)
        }
    }

    private fun stopVideoRecording() {
        activeRecording?.stop()
        activeRecording = null
        updateRecordingUi()
    }

    private fun toggleRecording() {
        if (activeRecording != null) {
            stopVideoRecording()
        } else {
            prepareRecordingWithBackend()
        }
    }

    private fun updateRecordingUi() {
        binding.recordButton.text = if (activeRecording != null) {
            getString(R.string.record_stop)
        } else {
            getString(R.string.record_start)
        }
    }

    private fun updateMovementState(tracked: TrackedPerson, nowMs: Long) {
        val box = tracked.trackingBox ?: tracked.colorComboBox
        if (tracked.mode == DetectionMode.LOST || box == null) {
            movementStatusText = getString(R.string.movement_unknown)
            return
        }
        val area = ((box[2] - box[0]) * (box[3] - box[1])).coerceAtLeast(1f)
        movementSamples += MovementSample(nowMs, area)
        while (movementSamples.isNotEmpty() && nowMs - movementSamples.first().timestampMs > MOVEMENT_WINDOW_MS) {
            movementSamples.removeFirst()
        }
        val oldest = movementSamples.firstOrNull()
        if (oldest == null || movementSamples.size < 3) {
            movementStatusText = getString(R.string.movement_collecting)
            return
        }
        val ratio = area / oldest.area
        movementStatusText = when {
            ratio >= APPROACH_RATIO_THRESHOLD -> getString(R.string.movement_forward)
            ratio <= RETREAT_RATIO_THRESHOLD -> getString(R.string.movement_backward)
            else -> getString(R.string.movement_stable)
        }
    }

    private fun currentMovementStatus(): String = movementStatusText.ifBlank {
        getString(R.string.movement_collecting)
    }

    private fun maybeSyncBackend(tracked: TrackedPerson) {
        if (!backendCoordinator.isSessionActive() || tracked.mode == DetectionMode.LOST) {
            return
        }
        sessionHeartbeatCounter += 1
        lifecycleScope.launch(Dispatchers.IO) {
            if (sessionHeartbeatCounter % HEARTBEAT_FRAME_INTERVAL == 0) {
                backendCoordinator.heartbeat(
                    trackStatus = tracked.mode.name,
                    fps = currentDisplayedFps(),
                    recording = activeRecording != null,
                    esp32Connected = esp32WifiConnector?.activeNetwork != null || esp32WifiConnector?.isAlreadyConnected() == true
                )
            }
        }
    }

    private fun appendStatusSuffix(base: String, suffix: String): String {
        return if (base.isBlank()) suffix else "$base | $suffix"
    }

    private fun prepareRecordingWithBackend() {
        if (!backendCoordinator.isSessionActive()) {
            binding.modelStatusText.text = getString(R.string.backend_session_failed, "session not started")
            return
        }
        binding.modelStatusText.text = appendStatusSuffix(
            binding.modelStatusText.text.toString(),
            getString(R.string.backend_quota_checking)
        )
        lifecycleScope.launch(Dispatchers.IO) {
            val summary = backendCoordinator.currentQuotaSummary().getOrElse { error ->
                withContext(Dispatchers.Main) {
                    binding.modelStatusText.text = getString(
                        R.string.backend_record_prepare_failed,
                        error.message ?: error.javaClass.simpleName
                    )
                }
                return@launch
            }
            backendQuotaText = getString(
                R.string.backend_quota_template,
                summary.quotaRemain,
                summary.quotaTotal
            )
            if (summary.quotaRemain <= 0) {
                withContext(Dispatchers.Main) {
                    binding.modelStatusText.text = getString(R.string.backend_quota_missing)
                }
                return@launch
            }
            val consume = backendCoordinator.consumeCurrentSessionQuotaIfNeeded().getOrElse { error ->
                withContext(Dispatchers.Main) {
                    binding.modelStatusText.text = getString(
                        R.string.backend_record_prepare_failed,
                        error.message ?: error.javaClass.simpleName
                    )
                }
                return@launch
            }
            if (!consume.success) {
                withContext(Dispatchers.Main) {
                    binding.modelStatusText.text = getString(R.string.backend_quota_missing)
                }
                return@launch
            }
            if (consume.quotaRemain >= 0) {
                backendQuotaText = getString(
                    R.string.backend_quota_remaining_simple,
                    consume.quotaRemain
                )
            }
            withContext(Dispatchers.Main) {
                binding.modelStatusText.text = appendStatusSuffix(
                    binding.modelStatusText.text.toString(),
                    getString(R.string.backend_record_ready)
                )
                startVideoRecording()
            }
        }
    }

    private fun applyEsp32PreviewScale() {
        val parentWidth = binding.root.width.takeIf { it > 0 } ?: resources.displayMetrics.widthPixels
        val parentHeight = binding.root.height.takeIf { it > 0 } ?: resources.displayMetrics.heightPixels
        val minWidth = dpToPx(180)
        val maxWidthByScreen = (parentWidth * 0.42f).toInt()
        val maxWidthByHeight = (parentHeight * 0.62f * 4f / 3f).toInt()
        val maxWidth = maxOf(minWidth + 1, minOf(maxWidthByScreen, maxWidthByHeight))
        val width = minWidth + ((maxWidth - minWidth) * (esp32PreviewScale / 100f)).toInt()
        val height = (width * 3f / 4f).toInt()
        val params = binding.esp32PreviewCard.layoutParams
        params.width = width
        params.height = height
        binding.esp32PreviewCard.layoutParams = params
        binding.esp32PreviewCard.requestLayout()
    }

    private fun dpToPx(dp: Int): Int {
        return (dp * resources.displayMetrics.density).toInt()
    }

    private fun getRecognitionNetworkStatus(): String {
        return if (esp32WifiConnector?.supportsConcurrentDeviceLink() == true) {
            getString(R.string.network_dual_ready)
        } else {
            getString(R.string.network_single_path)
        }
    }

    private fun refreshHomeSubtitle() {
        val primary = backendBootstrapText.ifBlank {
            getString(R.string.home_subtitle)
        }
        binding.homeSubtitleText.text = listOf(primary, backendQuotaText)
            .filter { it.isNotBlank() }
            .joinToString("\n")
    }

    private fun refreshHomeSubtitle(extra: String) {
        val items = mutableListOf<String>()
        val primary = backendBootstrapText.ifBlank { getString(R.string.home_subtitle) }
        if (primary.isNotBlank()) items += primary
        if (backendQuotaText.isNotBlank()) items += backendQuotaText
        if (extra.isNotBlank()) items += extra
        binding.homeSubtitleText.text = items.joinToString("\n")
    }

    private fun updateAuthUi(authenticated: Boolean) {
        binding.loginButton.isEnabled = !authenticated
        val dashboardVisible = authenticated && homeStage == HomeStage.DASHBOARD
        binding.logoutButton.visibility = if (dashboardVisible) View.VISIBLE else View.GONE
        binding.selectColorButton.isEnabled = dashboardVisible
        binding.startRecognitionButton.isEnabled = dashboardVisible
        binding.userInfoCard.visibility = if (authenticated && dashboardVisible) View.VISIBLE else View.GONE
        binding.loginPanel.visibility = if (!authenticated && homeStage == HomeStage.LOGIN) View.VISIBLE else View.GONE
        binding.registerPanel.visibility = if (!authenticated && homeStage == HomeStage.REGISTER) View.VISIBLE else View.GONE
        binding.provisionCard.visibility = if (homeStage == HomeStage.PROVISION) View.VISIBLE else View.GONE
        binding.homePrimaryModeChip.visibility = if (dashboardVisible) View.VISIBLE else View.GONE
        binding.selectColorButton.visibility = if (dashboardVisible) View.VISIBLE else View.GONE
        binding.startRecognitionButton.visibility = if (dashboardVisible) View.VISIBLE else View.GONE
        binding.homeFooterText.visibility = if (dashboardVisible) View.VISIBLE else View.GONE
        if (authenticated) {
            val profile = currentUserProfile
            binding.userNicknameText.text = getString(
                R.string.user_nickname_template,
                profile?.nickname ?: "--"
            )
            binding.userQuotaText.text = getString(
                R.string.user_quota_template,
                profile?.quotaRemain ?: 0,
                profile?.quotaTotal ?: 0
            )
            binding.userPlanText.text = getString(
                R.string.user_plan_template,
                profile?.planType ?: "--"
            )
            binding.userExpireText.text = getString(
                R.string.user_expire_template,
                profile?.planExpireAt ?: "--"
            )
        } else {
            binding.userNicknameText.text = getString(R.string.user_nickname_placeholder)
            binding.userQuotaText.text = getString(R.string.user_quota_placeholder)
            binding.userPlanText.text = getString(R.string.user_plan_placeholder)
            binding.userExpireText.text = getString(R.string.user_expire_placeholder)
        }
    }

    private fun showRegisterStage() {
        homeStage = HomeStage.REGISTER
        updateAuthUi(false)
    }

    private fun showLoginPanel() {
        homeStage = HomeStage.LOGIN
        updateAuthUi(false)
    }

    private fun showProvisionStage() {
        if (provisionReadyThisSession) {
            showLoginOrProvisionStage()
            return
        }
        // 配网页本身不应自动保留后台 BLE 扫描；只有明确进入自动配网流程时才继续持有它。
        if (!attemptingAutoProvision) {
            bleProvisioner.stop()
        }
        homeStage = HomeStage.PROVISION
        updateAuthUi(true)
    }

    private fun showDashboardStage() {
        homeStage = HomeStage.DASHBOARD
        updateAuthUi(true)
    }

    private fun showLoginOrProvisionStage() {
        if (provisionReadyThisSession) {
            if (!backendCoordinator.isAuthenticated()) {
                showLoginPanel()
            } else {
                showDashboardStage()
            }
            return
        }
        if (!hasProvisioningConfig()) {
            showProvisionStage()
        } else {
            if (!backendCoordinator.isAuthenticated()) {
                showLoginPanel()
            } else {
                showDashboardStage()
            }
            verifyProvisionStatusForHome()
        }
    }

    private fun verifyProvisionStatusForHome() {
        if (homeDeviceCheckRunning || screenMode != ScreenMode.HOME) return
        homeDeviceCheckRunning = true
        binding.homeSubtitleText.text = getString(R.string.esp32_discovery_wait_ip)
        lifecycleScope.launch(Dispatchers.IO) {
            val backendHost = fetchBackendLatestEsp32Host()
            val resolvedHost = backendHost?.let { resolveReachableEsp32HostWithRetries(it) }
            withContext(Dispatchers.Main) {
                homeDeviceCheckRunning = false
                if (screenMode != ScreenMode.HOME) return@withContext
                if (resolvedHost != null) {
                    markProvisionReady(resolvedHost)
                } else {
                    provisionReadyThisSession = false
                    showProvisionStage()
                    binding.homeSubtitleText.text = getString(R.string.esp32_home_reprovision_needed)
                }
            }
        }
    }

    private fun hasProvisioningConfig(): Boolean {
        val config = provisioningStore.load()
        return config.hotspotSsid.isNotBlank() &&
            config.hotspotPassword.isNotBlank() &&
            config.deviceNamePrefix.isNotBlank()
    }

    private fun clearRegisterForm() {
        binding.registerNicknameEditText.setText("")
        binding.registerAccountEditText.setText("")
        binding.registerPasswordEditText.setText("")
        binding.registerConfirmPasswordEditText.setText("")
    }

    private fun buildEsp32CaptureUrl(): String {
        return "http://$currentEsp32Host/capture"
    }

    private fun buildEsp32StreamUrl(): String {
        return "http://$currentEsp32Host:81/stream"
    }

    private fun isEsp32HostReachable(host: String): Boolean {
        val endpoints = listOf(
            "http://$host/capture",
            "http://$host/status"
        )
        return endpoints.any { endpoint ->
            runCatching {
                val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 1200
                    readTimeout = 1500
                    useCaches = false
                    setRequestProperty("Connection", "close")
                }
                try {
                    connection.connect()
                    connection.responseCode in 200..299
                } finally {
                    connection.disconnect()
                }
            }.getOrDefault(false)
        }
    }

    private fun isEsp32StatusReachable(host: String): Boolean {
        return runCatching {
            val connection = (URL("http://$host/status").openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 900
                readTimeout = 1100
                useCaches = false
                setRequestProperty("Connection", "close")
            }
            try {
                connection.connect()
                connection.responseCode in 200..299
            } finally {
                connection.disconnect()
            }
        }.getOrDefault(false)
    }

    private suspend fun resolveReachableEsp32Host(
        rememberedHost: String
    ): String? {
        updateEsp32DiscoveryStatus(getString(R.string.esp32_discovery_try_host, rememberedHost))
        if (isEsp32HostReachable(rememberedHost)) {
            return rememberedHost
        }
        updateEsp32DiscoveryStatus(getString(R.string.esp32_discovery_wait_ip))
        return null
    }

    private suspend fun fetchBackendLatestEsp32Host(): String? {
        if (!backendCoordinator.isAuthenticated()) return null
        val runtime = runCatching { backendCoordinator.getLatestDeviceRuntime().getOrThrow() }.getOrNull()
            ?: return null
        val ip = runtime.currentIp?.trim().orEmpty()
        if (ip.isBlank()) return null
        updateEsp32DiscoveryStatus(getString(R.string.esp32_backend_ip, ip))
        return ip
    }

    private suspend fun resolveReachableEsp32HostWithRetries(
        seedHost: String
    ): String? {
        repeat(ESP32_DIRECT_CONNECT_RETRY_COUNT) { attempt ->
            updateEsp32DiscoveryStatus(
                getString(
                    R.string.esp32_discovery_retry,
                    attempt + 1,
                    ESP32_DIRECT_CONNECT_RETRY_COUNT
                )
            )
            val resolved = if (seedHost == DEFAULT_ESP32_HOST) null else resolveReachableEsp32Host(seedHost)
            if (resolved != null) {
                return resolved
            }
            if (attempt < ESP32_DIRECT_CONNECT_RETRY_COUNT - 1) {
                delay(ESP32_DIRECT_CONNECT_RETRY_DELAY_MS)
            }
        }
        return null
    }

    private suspend fun waitForEsp32ServiceStartup(host: String): String? {
        updateEsp32DiscoveryStatus(getString(R.string.esp32_backend_ip, host))
        repeat(ESP32_SERVICE_STARTUP_WAIT_COUNT) { attempt ->
            updateEsp32DiscoveryStatus(
                getString(
                    R.string.esp32_service_starting_wait,
                    host,
                    attempt + 1,
                    ESP32_SERVICE_STARTUP_WAIT_COUNT
                )
            )
            // 同一地址分多轮探测，尽量覆盖 ESP32 刚拿到 IP、但 /status 还没起来的空窗。
            val resolved = resolveReachableEsp32HostWithRetries(host)
            if (resolved != null) {
                return resolved
            }
            if (attempt < ESP32_SERVICE_STARTUP_WAIT_COUNT - 1) {
                delay(ESP32_SERVICE_STARTUP_WAIT_DELAY_MS)
            }
        }
        return null
    }

    private fun updateEsp32DiscoveryStatus(text: String) {
        runOnUiThread {
            if (screenMode == ScreenMode.HOME) {
                binding.homeSubtitleText.text = text
            } else {
                binding.modelStatusText.text = appendStatusSuffix(text, getRecognitionNetworkStatus())
            }
        }
    }


    private fun currentDisplayedFps(): Float {
        val text = binding.fpsText.text?.toString().orEmpty()
        return text.substringAfter(": ", "0").toFloatOrNull() ?: 0f
    }

    private fun buildSessionSummaryForBackend(): String? {
        val summary = buildString {
            append(currentMovementStatus())
            val result = binding.resultText.text?.toString().orEmpty()
            if (result.isNotBlank()) {
                append(" | ")
                append(result.lineSequence().firstOrNull().orEmpty())
            }
        }.trim()
        return summary.ifBlank { null }
    }

    companion object {
        private const val INFERENCE_INTERVAL_MS = 120L
        private const val INIT_SECONDS = 10
        private const val ESP32_WIFI_SSID = "ESP32-CAM"
        private const val ESP32_WIFI_PASSWORD = ""
        private const val DEFAULT_ESP32_HOST = "192.168.4.1"
        private const val ESP32_POLL_INTERVAL_MS = 150L
        private const val ESP32_SOURCE_FAILURE_THRESHOLD = 4
        private const val ESP32_DIRECT_CONNECT_RETRY_COUNT = 6
        private const val ESP32_DIRECT_CONNECT_RETRY_DELAY_MS = 1200L
        private const val ESP32_SERVICE_STARTUP_WAIT_COUNT = 4
        private const val ESP32_SERVICE_STARTUP_WAIT_DELAY_MS = 1500L
        private const val ESP32_STATUS_PROBE_INTERVAL_MS = 1800L
        private const val ESP32_STATUS_PROBE_FAILURE_THRESHOLD = 3
        private const val PREFER_STREAM_SOURCE = true
        private const val MOVEMENT_WINDOW_MS = 1200L
        private const val APPROACH_RATIO_THRESHOLD = 1.12f
        private const val RETREAT_RATIO_THRESHOLD = 0.90f
        private const val HEARTBEAT_FRAME_INTERVAL = 15
    }
}
