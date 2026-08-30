package com.example.yolopose.network.backend

import android.content.Context

class SessionRepository(
    private val authRepository: AuthRepository,
    context: Context,
    private val client: SimpleBackendClient = SimpleBackendClient(context)
) {
    fun bindDevice(request: DeviceBindRequest): Result<Unit> = runCatching {
        val response = client.post(
            path = "/api/v1/device/bind",
            body = BackendJson.deviceBindRequestToJson(request),
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "device bind failed" } }
    }

    fun startSession(request: SessionStartRequest): Result<SessionStartResponse> = runCatching {
        val response = client.post(
            path = "/api/v1/session/start",
            body = BackendJson.sessionStartRequestToJson(request),
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "session start failed" } }
        BackendJson.parseSessionStartResponse(response.data ?: error("session start data missing"))
    }

    fun heartbeat(request: SessionHeartbeatRequest): Result<Unit> = runCatching {
        val response = client.post(
            path = "/api/v1/session/heartbeat",
            body = BackendJson.sessionHeartbeatRequestToJson(request),
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "heartbeat failed" } }
    }

    fun endSession(request: SessionEndRequest): Result<Unit> = runCatching {
        val response = client.post(
            path = "/api/v1/session/end",
            body = BackendJson.sessionEndRequestToJson(request),
            token = authRepository.token()
        )
        require(response.code == 0) { response.message.ifBlank { "session end failed" } }
    }
}
