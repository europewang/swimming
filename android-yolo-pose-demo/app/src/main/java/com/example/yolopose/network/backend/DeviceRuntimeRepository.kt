package com.example.yolopose.network.backend

import android.content.Context
import java.net.URLEncoder

class DeviceRuntimeRepository(
    private val authRepository: AuthRepository,
    context: Context,
    private val client: SimpleBackendClient = SimpleBackendClient(context)
) {
    fun getLatest(deviceCode: String): Result<DeviceRuntimeStatus> = runCatching {
        val encodedCode = URLEncoder.encode(deviceCode, Charsets.UTF_8.name())
        val response = client.get(
            path = "/api/v1/device/runtime/latest?device_code=$encodedCode",
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "device runtime latest failed" } }
        BackendJson.parseDeviceRuntimeStatus(response.data ?: error("device runtime data missing"))
    }
}
