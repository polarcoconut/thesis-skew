from api.train import restart
from app import app
from schema.job import Job
import sys, os, traceback, time, re

@app.celery.task(name='delete_temp_files')
def delete_temp_files():
    now = time.time()
    folder = 'temp_models'
    files_deleted = 0
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if re.search('*-*-*-*-*', f):
            if os.stat(f).st_mtime < now - 86400:
                if os.path.isfile(f):
                    os.remove(f)
                    files_deleted += 1
    print "Num files deleted:"
    print files_deleted

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
