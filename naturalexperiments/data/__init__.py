from .acic import load_acic
from .ihdp import load_ihdp
from .jobs import load_jobs
from .news import load_news
from .rorco import load_rorco, load_rorco_real
from .twins import load_twins

dataloaders = {
    'RORCO Real' : load_rorco_real,
    'RORCO' : load_rorco,
    'ACIC' : load_acic,
    'IHDP' : load_ihdp,
    'JOBS' : load_jobs,
    'NEWS' : load_news,
    'TWINS' : load_twins,
}