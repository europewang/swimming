package com.example.yolopose.network.backend

import android.content.Context

class AuthRepository(
    context: Context,
    private val client: SimpleBackendClient = SimpleBackendClient(context)
) {
    private val prefs = context.applicationContext.getSharedPreferences("backend_auth", Context.MODE_PRIVATE)

    fun login(account: String, password: String): Result<LoginResponse> = runCatching {
        val response = client.post(
            path = "/api/v1/auth/login",
            body = BackendJson.loginRequestToJson(LoginRequest(account = account, password = password))
        )
        require(response.code == 0) { response.message.ifBlank { "login failed" } }
        val data = response.data ?: error("login data missing")
        BackendJson.parseLoginResponse(data).also { login ->
            prefs.edit()
                .putString(KEY_ACCOUNT, account)
                .putString(KEY_TOKEN, login.token)
                .putLong(KEY_USER_ID, login.userId)
                .apply()
        }
    }

    fun register(account: String, password: String, nickname: String): Result<Unit> = runCatching {
        val response = client.post(
            path = "/api/v1/auth/register",
            body = BackendJson.registerRequestToJson(
                RegisterRequest(
                    account = account,
                    password = password,
                    nickname = nickname
                )
            )
        )
        require(response.code == 0) { response.message.ifBlank { "register failed" } }
    }

    fun getProfile(): Result<UserProfile> = runCatching {
        val response = client.get("/api/v1/auth/profile", token())
        require(response.code == 0) { response.message.ifBlank { "profile failed" } }
        BackendJson.parseUserProfile(response.data ?: error("profile data missing"))
    }

    fun savedAccount(): String? = prefs.getString(KEY_ACCOUNT, null)

    fun token(): String? = prefs.getString(KEY_TOKEN, null)

    fun hasToken(): Boolean = !token().isNullOrBlank()

    fun logout() {
        prefs.edit().clear().apply()
    }

    companion object {
        private const val KEY_ACCOUNT = "account"
        private const val KEY_TOKEN = "token"
        private const val KEY_USER_ID = "user_id"
    }
}
