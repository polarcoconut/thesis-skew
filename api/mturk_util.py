from app import app
from boto.mturk.question import ExternalQuestion
from boto.mturk.price import Price
from boto.mturk.qualification import Qualifications
from boto.mturk.qualification import PercentAssignmentsApprovedRequirement
from boto.mturk.qualification import NumberHitsApprovedRequirement
from boto.mturk.qualification import LocaleRequirement
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import HTMLQuestion
from boto.mturk.connection import MTurkRequestError
import sys
import datetime

def delete_hits(hits_to_delete):
    print "Connecting to Turk host at"
    print app.config['MTURK_HOST']
    sys.stdout.flush()
    
    mturk = MTurkConnection(app.config['AWS_ACCESS_KEY_ID'],
                            app.config['AWS_SECRET_ACCESS_KEY'],
                            host=app.config['MTURK_HOST'])

    print "Deleting extra hits"
    
    for hit in hits_to_delete:
        try:
            mturk.disable_hit(hit)
        except MTurkRequestError:
            print "Trying to delete hit that doesn't exist"
        
    return True

#task_id is the crowd_js assigned task id.
def create_hits(category_id, task_id, num_hits):

    print "Connecting to Turk host at"
    print app.config['MTURK_HOST']
    sys.stdout.flush()
    
    mturk = MTurkConnection(app.config['AWS_ACCESS_KEY_ID'],
                            app.config['AWS_SECRET_ACCESS_KEY'],
                            host=app.config['MTURK_HOST'])

    print "Uploading %d hits to turk" % num_hits
    hits = []
    qualifications = app.config['QUALIFICATIONS']

    for hit_num in range(num_hits):

        """
        layout_params = LayoutParameters()

        #layout_params.add(
        #    LayoutParameter('questionNumber', '%s' % hit_num))
        layout_params.add(
            LayoutParameter('task_id', '%s' % task_id))
        layout_params.add(
            LayoutParameter('requester_id', '%s'%
                            app.config['CROWDJS_REQUESTER_ID']))
        layout_params.add(
            LayoutParameter(
                'task_data_url', '%s'%
                app.config['CROWDJS_GET_TASK_DATA_URL']))
        layout_params.add(
            LayoutParameter(
                'submit_answer_url', '%s'%
                app.config['CROWDJS_SUBMIT_ANSWER_URL']))
        layout_params.add(
            LayoutParameter(
                'compute_taboo_url', '%s'%
                app.config['SUBMIT_TABOO_URL']))
        layout_params.add(
            LayoutParameter(
                'return_hit_url', '%s'%
                app.config['CROWDJS_RETURN_HIT_URL']))
        layout_params.add(
            LayoutParameter(
                'assign_url', '%s'%
                app.config['CROWDJS_ASSIGN_URL']))

        layout_params.add(
            LayoutParameter(
                'taboo_threshold', '%s'%
                app.config['TABOO_THRESHOLD']))        
        print layout_params
        sys.stdout.flush()
        
        #hit = mturk.create_hit(
        #    hit_type= hit_type_id,
        #    hit_layout = hit_layout_id,
        #    layout_params = layout_params)[0]
        """

        category = app.config['EXAMPLE_CATEGORIES'][category_id]
        hit_html = category['hit_html']
        hit_html = hit_html.replace('${task_id}', task_id)
        hit_html = hit_html.replace('${requester_id}',
                                    app.config['CROWDJS_REQUESTER_ID'])
        hit_html = hit_html.replace('${task_data_url}',
                                    app.config['CROWDJS_GET_TASK_DATA_URL'])
        hit_html = hit_html.replace('${submit_answer_url}',
                                    app.config['CROWDJS_SUBMIT_ANSWER_URL'])
        hit_html = hit_html.replace('${compute_taboo_url}',
                                    app.config['SUBMIT_TABOO_URL'])
        hit_html = hit_html.replace('${return_hit_url}',
                                    app.config['CROWDJS_RETURN_HIT_URL'])
        hit_html = hit_html.replace('${assign_url}',
                                    app.config['CROWDJS_ASSIGN_URL'])
        hit_html = hit_html.replace('${taboo_threshold}',
                                    str(app.config['TABOO_THRESHOLD']))
        hit_html = hit_html.replace('${mturk_external_submit}',
                                    str(app.config['MTURK_EXTERNAL_SUBMIT']))

        
        question = HTMLQuestion(hit_html, 800)
        hit = mturk.create_hit(
            title = category['task_name'],
            description = category['task_description'],
            question = question,
            reward = Price(category['price']),
            duration = datetime.timedelta(minutes=10),
            lifetime = datetime.timedelta(days=7),
            keywords = 'information extraction, events, natural language processing',
            max_assignments = app.config['CONTROLLER_APQ'],
            approval_delay = 3600,
            qualifications = qualifications)[0]

        hits.append(hit.HITId)
        
    return hits
