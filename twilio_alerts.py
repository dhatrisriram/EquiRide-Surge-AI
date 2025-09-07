from dotenv import load_dotenv
import os
from twilio.rest import Client

load_dotenv()
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE = os.getenv('TWILIO_PHONE')

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_sms(to_number, message):
    client.messages.create(body=message, from_=TWILIO_PHONE, to=to_number)

def send_whatsapp(to_number, message):
    client.messages.create(body=message, from_='whatsapp:'+TWILIO_PHONE, to='whatsapp:'+to_number)

# Example usage
if __name__ == "__main__":
    send_sms("+919876543210", "🚖 Surge alert: High demand in Zone A!")
    send_whatsapp("+919876543210", "🚖 Surge alert: High demand in Zone A!")
