from api.train import restart
from app import app
from schema.job import Job

@app.celery.task(name='run_gather')
def run_gather():

    jobs_checked = 0
    jobs = Job.objects(status='Running')
    for job in jobs:
        try:
            if job.lock:
                print"job is locked. go to next job"
                continue
        except AttributeError:
            print "job doesn't have lock property"
            
        jobs_checked += 1
        print "Running Gather for job %s" % job.id
        job.lock = True
        job.save()
        restart(job.id)
        job.lock = False
        job.save()
        
    print "%d jobs restarted" % jobs_checked
    
    return True