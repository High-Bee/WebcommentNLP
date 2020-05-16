import datetime
from pytz import timezone, utc

def KST_now():
    KST = timezone('Asia/Seoul')
    now = datetime.datetime.utcnow()
    KST_now = utc.localize(now).astimezone(KST)
    return KST_now.strftime("KST %Y-%m-%d %H:%M")

