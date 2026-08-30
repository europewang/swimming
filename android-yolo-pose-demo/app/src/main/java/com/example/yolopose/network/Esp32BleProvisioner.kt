package com.example.yolopose.network

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import java.nio.charset.Charset
import java.util.UUID

class Esp32BleProvisioner(
    context: Context
) {
    private val appContext = context.applicationContext
    private val bluetoothManager = appContext.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val mainHandler = Handler(Looper.getMainLooper())

    private var scanner: BluetoothLeScanner? = null
    private var gatt: BluetoothGatt? = null
    private var callback: ((Esp32ProvisioningState) -> Unit)? = null
    private var config: Esp32ProvisioningConfig? = null
    private var connectedDeviceName: String? = null
    private var discoverRetryCount = 0
    private var pendingNotificationEnable = false
    private var completionReached = false
    private var stoppingManually = false
    private var deviceAcknowledgedProvisioning = false

    private val timeoutRunnable = Runnable {
        log("timeout fired")
        emit(Esp32ProvisioningState.Failed("蓝牙配网超时，请确认 ESP32 已开启 BLE 配网固件"))
        stop()
    }

    @SuppressLint("MissingPermission")
    fun startProvisioning(
        provisioningConfig: Esp32ProvisioningConfig,
        stateCallback: (Esp32ProvisioningState) -> Unit
    ) {
        log("startProvisioning ssid=${provisioningConfig.hotspotSsid}, prefix=${provisioningConfig.deviceNamePrefix}")
        stop()
        callback = stateCallback
        config = provisioningConfig
        completionReached = false
        stoppingManually = false
        deviceAcknowledgedProvisioning = false

        val adapter = bluetoothManager.adapter
        if (adapter == null || !adapter.isEnabled) {
            log("bluetooth adapter unavailable or disabled")
            emit(Esp32ProvisioningState.Failed("蓝牙未开启，无法执行自动配网"))
            return
        }

        scanner = adapter.bluetoothLeScanner
        if (scanner == null) {
            log("bluetoothLeScanner unavailable")
            emit(Esp32ProvisioningState.Failed("当前设备不支持 BLE 扫描"))
            return
        }

        emit(Esp32ProvisioningState.Scanning)
        scheduleTimeout(SCAN_TIMEOUT_MS)
        log("startScan")
        scanner?.startScan(scanCallback)
    }

    @SuppressLint("MissingPermission")
    fun stop() {
        log("stop provisioning")
        stoppingManually = true
        mainHandler.removeCallbacks(timeoutRunnable)
        runCatching { scanner?.stopScan(scanCallback) }
        scanner = null
        runCatching { gatt?.close() }
        gatt = null
        connectedDeviceName = null
        discoverRetryCount = 0
        pendingNotificationEnable = false
        mainHandler.post {
            stoppingManually = false
        }
    }

    @SuppressLint("MissingPermission")
    private fun connect(device: BluetoothDevice) {
        runCatching { scanner?.stopScan(scanCallback) }
        connectedDeviceName = device.name ?: device.address
        log("connect device=${connectedDeviceName} address=${device.address}")
        emit(Esp32ProvisioningState.DeviceFound(connectedDeviceName.orEmpty()))
        scheduleTimeout(CONNECT_TIMEOUT_MS)
        gatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            device.connectGatt(appContext, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
        } else {
            device.connectGatt(appContext, false, gattCallback)
        }
    }

    @SuppressLint("MissingPermission")
    private fun sendProvisioningPayload(gatt: BluetoothGatt) {
        val currentConfig = config ?: return
        val service = gatt.getService(SERVICE_UUID)
        val writeCharacteristic = service?.getCharacteristic(PROVISION_WRITE_UUID)
        val notifyCharacteristic = service?.getCharacteristic(PROVISION_NOTIFY_UUID)
        if (service == null || writeCharacteristic == null || notifyCharacteristic == null) {
            log(
                "service/characteristic missing service=${service != null} " +
                    "write=${writeCharacteristic != null} notify=${notifyCharacteristic != null}"
            )
            retryDiscoverServicesOrFail(gatt)
            return
        }

        emit(Esp32ProvisioningState.ServicesReady)
        if (enableNotifications(gatt, notifyCharacteristic)) {
            log("descriptor write started, wait for onDescriptorWrite before sending payload")
            pendingNotificationEnable = true
            scheduleTimeout(WRITE_TIMEOUT_MS)
        } else {
            log("notification descriptor missing or write failed, send payload directly")
            writeProvisioningPayload(gatt, writeCharacteristic, currentConfig)
        }
    }

    @SuppressLint("MissingPermission")
    private fun enableNotifications(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic): Boolean {
        val notificationEnabled = gatt.setCharacteristicNotification(characteristic, true)
        log("setCharacteristicNotification=${notificationEnabled} uuid=${characteristic.uuid}")
        val descriptor = characteristic.getDescriptor(CLIENT_CONFIG_UUID) ?: return false
        descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
        val started = gatt.writeDescriptor(descriptor)
        log("writeDescriptor started=${started} descriptor=${descriptor.uuid}")
        return started
    }

    @SuppressLint("MissingPermission")
    private fun writeProvisioningPayload(
        gatt: BluetoothGatt,
        writeCharacteristic: BluetoothGattCharacteristic,
        currentConfig: Esp32ProvisioningConfig
    ) {
        emit(Esp32ProvisioningState.SendingHotspotConfig(currentConfig.hotspotSsid))
        scheduleTimeout(WRITE_TIMEOUT_MS)

        val payloadText = JSONObject().apply {
            put("cmd", "provision_wifi")
            put("mode", "phone_hotspot")
            put("ssid", currentConfig.hotspotSsid)
            put("password", currentConfig.hotspotPassword)
        }.toString()
        log("write payload=$payloadText")
        val payload = payloadText.toByteArray(Charset.forName("UTF-8"))

        writeCharacteristic.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
        writeCharacteristic.value = payload
        val writeStarted = gatt.writeCharacteristic(writeCharacteristic)
        log("writeCharacteristic started=${writeStarted} uuid=${writeCharacteristic.uuid}")
        if (!writeStarted) {
            emit(Esp32ProvisioningState.Failed("热点配置发送失败"))
            stop()
            return
        }
        emit(Esp32ProvisioningState.WaitingForDeviceJoin(currentConfig.hotspotSsid))
    }

    private fun handleDeviceResponse(raw: ByteArray?) {
        val text = raw?.toString(Charset.forName("UTF-8")).orEmpty()
        log("device response raw=$text")
        if (text.isBlank()) return

        val normalized = normalizeJsonPayload(text)
        log("device response normalized=$normalized")

        val payload = runCatching { JSONObject(normalized) }.getOrElse { error ->
            val heuristicStatus = extractStatusHeuristically(normalized)
            log("device response parse failed=${error.message}, heuristicStatus=$heuristicStatus")
            when (heuristicStatus) {
                "saved" -> {
                    deviceAcknowledgedProvisioning = true
                    emit(Esp32ProvisioningState.SendingHotspotConfig(config?.hotspotSsid.orEmpty()))
                    return
                }
                "connecting", "sta_connected" -> {
                    deviceAcknowledgedProvisioning = true
                    emit(Esp32ProvisioningState.WaitingForDeviceJoin(config?.hotspotSsid.orEmpty()))
                    return
                }
                "ready" -> {
                    deviceAcknowledgedProvisioning = true
                    emit(Esp32ProvisioningState.WaitingForDeviceJoin(config?.hotspotSsid.orEmpty()))
                    return
                }
                "error" -> {
                    emit(Esp32ProvisioningState.Failed("ESP32 返回错误状态，但消息格式异常"))
                    stop()
                    return
                }
                else -> return
            }
        }

        val status = payload.optString("status")
        log("device response status=$status")
        when (status) {
            "saved" -> {
                deviceAcknowledgedProvisioning = true
                emit(Esp32ProvisioningState.SendingHotspotConfig(payload.optString("ssid").ifBlank {
                    config?.hotspotSsid.orEmpty()
                }))
            }
            "connecting", "sta_connected" -> {
                deviceAcknowledgedProvisioning = true
                emit(Esp32ProvisioningState.WaitingForDeviceJoin(payload.optString("ssid").ifBlank {
                    config?.hotspotSsid.orEmpty()
                }))
            }
            "ready" -> {
                deviceAcknowledgedProvisioning = true
                val host = payload.optString("ip")
                    .ifBlank { payload.optString("host") }
                if (host.isBlank()) {
                    emit(Esp32ProvisioningState.Failed("ESP32 已就绪，但未返回可用地址"))
                    stop()
                    return
                }
                completionReached = true
                emit(Esp32ProvisioningState.Completed(host))
                stop()
            }
            "error" -> {
                val message = payload.optString("message").ifBlank { "ESP32 配网失败" }
                emit(Esp32ProvisioningState.Failed(message))
                stop()
            }
            else -> {
                log("ignore non-terminal status=$status")
            }
        }
    }

    private fun emit(state: Esp32ProvisioningState) {
        mainHandler.post { callback?.invoke(state) }
    }

    private fun scheduleTimeout(durationMs: Long) {
        mainHandler.removeCallbacks(timeoutRunnable)
        mainHandler.postDelayed(timeoutRunnable, durationMs)
        log("scheduleTimeout ${durationMs}ms")
    }

    @SuppressLint("MissingPermission")
    private fun retryDiscoverServicesOrFail(gatt: BluetoothGatt) {
        if (discoverRetryCount < MAX_DISCOVER_RETRIES) {
            discoverRetryCount += 1
            log("retry discoverServices count=$discoverRetryCount")
            emit(Esp32ProvisioningState.DiscoveringServices)
            scheduleTimeout(DISCOVER_TIMEOUT_MS)
            mainHandler.postDelayed({
                val started = runCatching { gatt.discoverServices() }.getOrDefault(false)
                log("discoverServices retry started=$started")
            }, DISCOVER_RETRY_DELAY_MS)
            return
        }
        log("discoverServices exhausted retries")
        emit(Esp32ProvisioningState.Failed("ESP32 BLE 配网服务未找到，请刷入支持自动配网的固件"))
        stop()
    }

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val currentConfig = config ?: return
            val name = result.device.name ?: result.scanRecord?.deviceName ?: return
            log("scan result name=$name address=${result.device.address}")
            if (name.startsWith(currentConfig.deviceNamePrefix, ignoreCase = true)) {
                connect(result.device)
            }
        }

        override fun onScanFailed(errorCode: Int) {
            log("scan failed code=$errorCode")
            emit(Esp32ProvisioningState.Failed("BLE 扫描失败，错误码: $errorCode"))
            stop()
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            log("onConnectionStateChange status=$status newState=$newState")
            if (completionReached && (status == 19 || newState == BluetoothGatt.STATE_DISCONNECTED)) {
                log("ignore disconnect after completion status=$status newState=$newState")
                return
            }
            if (stoppingManually && newState == BluetoothGatt.STATE_DISCONNECTED) {
                log("ignore disconnect caused by local stop")
                return
            }
            if (status != BluetoothGatt.GATT_SUCCESS) {
                if (status == 19) {
                    log("peer terminated BLE connection before Android considered flow complete")
                }
                emit(Esp32ProvisioningState.Failed("BLE 连接状态异常: $status"))
                stop()
                return
            }
            if (newState == BluetoothGatt.STATE_CONNECTED) {
                emit(Esp32ProvisioningState.DeviceConnected(connectedDeviceName.orEmpty()))
                emit(Esp32ProvisioningState.DiscoveringServices)
                scheduleTimeout(DISCOVER_TIMEOUT_MS)
                mainHandler.postDelayed({
                    val started = runCatching { gatt.discoverServices() }.getOrDefault(false)
                    log("discoverServices started=$started")
                }, DISCOVER_RETRY_DELAY_MS)
            } else if (newState == BluetoothGatt.STATE_DISCONNECTED) {
                emit(Esp32ProvisioningState.Failed("ESP32 BLE 连接已断开"))
                stop()
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            val serviceSummary = gatt.services.joinToString { it.uuid.toString() }
            log("onServicesDiscovered status=$status services=$serviceSummary")
            if (status == BluetoothGatt.GATT_SUCCESS) {
                discoverRetryCount = 0
                sendProvisioningPayload(gatt)
            } else {
                retryDiscoverServicesOrFail(gatt)
            }
        }

        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: BluetoothGattDescriptor,
            status: Int
        ) {
            log("onDescriptorWrite uuid=${descriptor.uuid} status=$status pendingNotificationEnable=$pendingNotificationEnable")
            if (!pendingNotificationEnable) return
            pendingNotificationEnable = false
            if (status != BluetoothGatt.GATT_SUCCESS) {
                emit(Esp32ProvisioningState.Failed("ESP32 通知通道开启失败"))
                stop()
                return
            }
            val currentConfig = config ?: return
            val service = gatt.getService(SERVICE_UUID)
            val writeCharacteristic = service?.getCharacteristic(PROVISION_WRITE_UUID)
            if (writeCharacteristic == null) {
                emit(Esp32ProvisioningState.Failed("热点配置通道不存在"))
                stop()
                return
            }
            writeProvisioningPayload(gatt, writeCharacteristic, currentConfig)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            if (characteristic.uuid == PROVISION_NOTIFY_UUID) {
                log("onCharacteristicChanged notify uuid=${characteristic.uuid}")
                handleDeviceResponse(characteristic.value)
            }
        }

        override fun onCharacteristicWrite(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            status: Int
        ) {
            log("onCharacteristicWrite uuid=${characteristic.uuid} status=$status")
            if (status != BluetoothGatt.GATT_SUCCESS) {
                if (deviceAcknowledgedProvisioning || completionReached) {
                    log("ignore characteristic write failure after device already acknowledged provisioning")
                    return
                }
                emit(Esp32ProvisioningState.Failed("热点配置写入失败"))
                stop()
            }
        }
    }

    private fun log(message: String) {
        Log.d(TAG, message)
    }

    private fun normalizeJsonPayload(raw: String): String {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}')
        if (start >= 0 && end > start) {
            return raw.substring(start, end + 1)
        }
        return raw.trim().removePrefix("(")
    }

    private fun extractStatusHeuristically(raw: String): String? {
        return Regex("\"status\"\\s*:\\s*\"([^\"]+)\"")
            .find(raw)
            ?.groupValues
            ?.getOrNull(1)
    }

    companion object {
        private const val TAG = "Esp32BleProvisioner"
        val SERVICE_UUID: UUID = UUID.fromString("0000FF10-0000-1000-8000-00805F9B34FB")
        val PROVISION_WRITE_UUID: UUID = UUID.fromString("0000FF11-0000-1000-8000-00805F9B34FB")
        val PROVISION_NOTIFY_UUID: UUID = UUID.fromString("0000FF12-0000-1000-8000-00805F9B34FB")
        private val CLIENT_CONFIG_UUID: UUID = UUID.fromString("00002902-0000-1000-8000-00805F9B34FB")
        private const val SCAN_TIMEOUT_MS = 18_000L
        private const val CONNECT_TIMEOUT_MS = 15_000L
        private const val DISCOVER_TIMEOUT_MS = 12_000L
        private const val WRITE_TIMEOUT_MS = 20_000L
        private const val DISCOVER_RETRY_DELAY_MS = 600L
        private const val MAX_DISCOVER_RETRIES = 2
    }
}
