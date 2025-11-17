# src/viz/alerts_stub.py
def send_alert_stub(twilio_sid, twilio_token, from_number, to_number, message, media_url=None):
    """
    Demo stub: logs the alert payload and returns success.
    Replace with real Twilio client for production.
    """
    print("=== ALERT STUB ===")
    print("TO:", to_number)
    print("FROM:", from_number)
    print("BODY:", message)
    if media_url:
        print("MEDIA:", media_url)
    print("==================")
    return True, {"status":"stub-sent"}
