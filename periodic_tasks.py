from api.train import restart
from app import app
from schema.job import Job
import sys, os, traceback, time, re
import pprint
import inspect
import psutil

#@app.celery.task(name='delete_temp_files')
#def delete_temp_files():
#    now = time.time()
#    folder = 'temp_models'
#    files_deleted = 0
#    for f in os.listdir(folder):
#        f = os.path.join(folder, f)
#        if re.search('*-*-*-*-*', f):
#            if os.stat(f).st_mtime < now - 86400:
#                if os.path.isfile(f):
#                    os.remove(f)
#                    files_deleted += 1
#    print "Num files deleted:"
#    print files_deleted

@app.celery.task(name='run_gather')
def run_gather():

    jobs_checked = 0
    jobs = Job.objects(status='Running')

    print "%d jobs currently running" % len(jobs)
    
    for job in jobs:
        #job.status = 'Finished'
        #job.save()
        #continue

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
        #if True:
            try:
                jobs_checked += 1
                print "Running Gather for job %s" % job.id
                restart(str(job.id))                
            except Exception:
                print "Exception:"
                print '-'*60
                traceback.print_exc(file=sys.stdout)
                job.exceptions.append(traceback.format_exc())
                
                local_variables = inspect.trace()[-1][0].f_locals
                pprint.pprint(local_variables)
                job.exceptions.append(pprint.pformat(local_variables))
                job.save()
                print '-'*60
                print "Killing background processes"
                
                celery_processes = []
                for proc in psutil.process_iter():
                    try:
                        if proc.name() == "celery":
                            celery_processes.append(proc)
                    except psutil.NoSuchProcess:
                        pass
        
                celery_processes = sorted(celery_processes,
                                          key= lambda proc: proc.create_time,
                                          reverse=True)

                print [(proc.name(), 
                        proc.create_time) for proc in celery_processes]
        
                for proc in celery_processes:
                    if proc.pid == os.getpid():
                        continue
                    proc.kill()
            finally:
                release_lock()
                        
        else:
            print"job is locked. go to next job"
                
    print "%d jobs restarted" % jobs_checked
    
    return True
