package com.example.yolopose.network.backend

import android.content.Context
import com.example.yolopose.BuildConfig
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class BackendCoordinator(context: Context) {
    private val appContext = context.applicationContext
    private val authRepository = AuthRepository(appContext)
    private val quotaRepository = QuotaRepository(authRepository, appContext)
    private val sessionRepository = SessionRepository(authRepository, appContext)
    private val deviceRuntimeRepository = DeviceRuntimeRepository(authRepository, appContext)

    private var activeSessionId: String? = null
    private var activeSessionStartedAtMs: Long = 0L
    private var quotaConsumed = false

    fun bootstrap(): Result<BackendBootstrapState> = runCatching {
        require(authRepository.hasToken()) { "请先登录" }
        val profile = authRepository.getProfile().getOrThrow()
        val quota = quotaRepository.getSummary().getOrThrow()
        BackendBootstrapState(
            token = authRepository.token(),
            profile = profile,
            quotaSummary = quota
        )
    }

    fun login(account: String, password: String): Result<BackendBootstrapState> = runCatching {
        authRepository.login(account, password).getOrThrow()
        bootstrap().getOrThrow()
    }

    fun register(account: String, password: String, nickname: String): Result<Unit> {
        return authRepository.register(account, password, nickname)
    }

    fun logout() {
        authRepository.logout()
    }

    fun isAuthenticated(): Boolean = authRepository.hasToken()

    fun savedAccount(): String? = authRepository.savedAccount()

    fun bindDefaultDevice(): Result<Unit> {
        return sessionRepository.bindDevice(
            DeviceBindRequest(
                deviceCode = DEFAULT_DEVICE_CODE,
                deviceName = DEFAULT_DEVICE_NAME,
                wifiSsid = DEFAULT_WIFI_SSID
            )
        )
    }

    fun getLatestDeviceRuntime(): Result<DeviceRuntimeStatus> {
        return deviceRuntimeRepository.getLatest(DEFAULT_DEVICE_CODE)
    }

    fun startTrackingSession(mode: String, videoSource: String): Result<String> = runCatching {
        val sessionId = buildSessionId()
        val response = sessionRepository.startSession(
            SessionStartRequest(
                sessionId = sessionId,
                deviceCode = DEFAULT_DEVICE_CODE,
                mode = mode,
                videoSource = videoSource,
                appVersion = BuildConfig.VERSION_NAME
            )
        ).getOrThrow()
        activeSessionId = response.sessionId.ifBlank { sessionId }
        activeSessionStartedAtMs = System.currentTimeMillis()
        quotaConsumed = false
        activeSessionId ?: sessionId
    }

    fun consumeCurrentSessionQuotaIfNeeded(): Result<ConsumeQuotaResponse> = runCatching {
        val sessionId = activeSessionId ?: error("session not started")
        if (quotaConsumed) {
            return@runCatching ConsumeQuotaResponse(success = true, quotaRemain = -1)
        }
        quotaRepository.consume(
            sessionId = sessionId,
            consumeType = "swim_track",
            consumeCount = 1
        ).getOrThrow().also {
            quotaConsumed = true
        }
    }

    fun currentQuotaSummary(): Result<QuotaSummary> {
        return quotaRepository.getSummary()
    }

    fun heartbeat(trackStatus: String, fps: Float, recording: Boolean, esp32Connected: Boolean): Result<Unit> {
        val sessionId = activeSessionId ?: return Result.success(Unit)
        return sessionRepository.heartbeat(
            SessionHeartbeatRequest(
                sessionId = sessionId,
                trackStatus = trackStatus,
                fps = fps,
                recording = recording,
                esp32Connected = esp32Connected
            )
        )
    }

    fun endActiveSession(avgFps: Float, resultSummary: String?): Result<Unit> {
        val sessionId = activeSessionId ?: return Result.success(Unit)
        val durationSec = (((System.currentTimeMillis() - activeSessionStartedAtMs) / 1000L).toInt()).coerceAtLeast(0)
        val result = sessionRepository.endSession(
            SessionEndRequest(
                sessionId = sessionId,
                durationSec = durationSec,
                avgFps = avgFps,
                recordFilePath = null,
                resultSummary = resultSummary
            )
        )
        activeSessionId = null
        activeSessionStartedAtMs = 0L
        quotaConsumed = false
        return result
    }

    fun isSessionActive(): Boolean = !activeSessionId.isNullOrBlank()

    private fun buildSessionId(): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return "session_${timestamp}_${System.currentTimeMillis() % 1000}"
    }

    companion object {
        private const val DEFAULT_DEVICE_CODE = "esp32cam-001"
        private const val DEFAULT_DEVICE_NAME = "Pool Cam 1"
        private const val DEFAULT_WIFI_SSID = "ESP32-CAM"
    }
}
