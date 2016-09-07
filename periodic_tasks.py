from api.train import restart
from app import app
from schema.job import Job
import sys, os, traceback
import redis

@app.celery.task(name='run_gather')
def run_gather():

    jobs_checked = 0
    jobs = Job.objects(status='Running')

    for job in jobs:
        #FOR DEBUGGING PURPOSES
        #job.lock = False
        #job.save()
        #raise Exception

        if len(job.checkpoints.keys()) == 0:
            continue

        lock_key = job.id
        redis_handle = redis.Redis.from_url(app.config['REDIS_URL'])
        
        acquire_lock = lambda: redis_handle.setnx(lock_key, '1')
        release_lock = lambda: redis_handle.delete(lock_key)

        if acquire_lock():
            try:
                jobs_checked += 1
                print "Running Gather for job %s" % job.id
                restart(job.id)                
            except Exception:
                print "Exception:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                print '-'*60
            finally:
                release_lock()
                        
        else:
            print"job is locked. go to next job"
                
    print "%d jobs restarted" % jobs_checked
    
    return True
