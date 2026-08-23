import pilot

# VERIFIED_ZERO trigger: Gemini 3.6 Flash real-provider pilot
pilot.MODEL = "gemini-3.6-flash"
pilot.ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{pilot.MODEL}:generateContent"
pilot.main()
