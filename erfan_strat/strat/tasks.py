from celery import shared_task

'celery -A mysite worker --concurrency=2 --without-gossip --pool=solo'

'celery -A mysite beat -l info'


@shared_task
def cryptoForex(x=3,y=5):
    print( x*y)