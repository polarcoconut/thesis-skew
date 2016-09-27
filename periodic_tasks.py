from api.train import restart
from app import app
from schema.job import Job
import sys, os, traceback

@app.celery.task(name='run_gather')
def run_gather():

    jobs_checked = 0
    jobs = Job.objects(status='Running')

    print "%d jobs currently running" % len(jobs)
    
    for job in jobs:
        lock_key = job.id
        
        acquire_lock = lambda: app.redis.setnx(lock_key, '1')
        release_lock = lambda: app.redis.delete(lock_key)

        #FOR DEBUGGING PURPOSES
        #release_lock()
        #print "LOCK RELEASED"
        #raise Exception

        if len(job.checkpoints.keys()) == 0:
            continue


        if acquire_lock():
            try:
                jobs_checked += 1
                print "Running Gather for job %s" % job.id
                restart(str(job.id))                
            except Exception:
                print "Exception:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                job.exceptions.append(traceback.format_exc())
                job.save()
                print '-'*60
            finally:
                release_lock()
                        
        else:
            print"job is locked. go to next job"
                
    print "%d jobs restarted" % jobs_checked
    
    return True
