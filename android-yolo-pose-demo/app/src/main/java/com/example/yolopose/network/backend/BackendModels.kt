package com.example.yolopose.network.backend

data class ApiEnvelope<T>(
    val code: Int,
    val message: String,
    val data: T? = null
)

data class LoginRequest(
    val loginType: String = "password",
    val account: String,
    val password: String
)

data class RegisterRequest(
    val account: String,
    val password: String,
    val nickname: String
)

data class LoginResponse(
    val userId: Long,
    val token: String,
    val nickname: String,
    val quotaRemain: Int,
    val planType: String
)

data class UserProfile(
    val userId: Long,
    val nickname: String,
    val avatar: String?,
    val quotaRemain: Int,
    val quotaTotal: Int,
    val planType: String,
    val planExpireAt: String?
)

data class QuotaSummary(
    val quotaTotal: Int,
    val quotaUsed: Int,
    val quotaRemain: Int,
    val resetRule: String?
)

data class DeviceBindRequest(
    val deviceCode: String,
    val deviceName: String,
    val wifiSsid: String,
    val macAddress: String? = null
)

data class SessionStartRequest(
    val sessionId: String,
    val deviceCode: String,
    val mode: String,
    val videoSource: String,
    val appVersion: String
)

data class SessionStartResponse(
    val sessionId: String,
    val accepted: Boolean
)

data class ConsumeQuotaRequest(
    val sessionId: String,
    val consumeType: String,
    val consumeCount: Int
)

data class ConsumeQuotaResponse(
    val success: Boolean,
    val quotaRemain: Int
)

data class SessionHeartbeatRequest(
    val sessionId: String,
    val trackStatus: String,
    val fps: Float,
    val recording: Boolean,
    val esp32Connected: Boolean
)

data class SessionEndRequest(
    val sessionId: String,
    val durationSec: Int,
    val avgFps: Float,
    val recordFilePath: String?,
    val resultSummary: String?
)

data class BackendBootstrapState(
    val token: String?,
    val profile: UserProfile?,
    val quotaSummary: QuotaSummary?
)

data class DeviceRuntimeStatus(
    val deviceCode: String,
    val currentIp: String?,
    val currentHttpPort: Int,
    val currentStreamPort: Int,
    val onlineStatus: String,
    val ssid: String?,
    val lastSeenAt: String?
)
