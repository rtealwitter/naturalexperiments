from .acic import load_acic16, load_acic17
from .ihdp import load_ihdp
from .jobs import load_jobs
from .news import load_news
from .rorco import load_rorco, load_rorco_real
from .twins import load_twins

dataloaders = {
    'ACIC 2016' : load_acic16,
    'ACIC 2017' : load_acic17,
    'IHDP' : load_ihdp,
    'JOBS' : load_jobs,
    'NEWS' : load_news,
    'TWINS' : load_twins,
    'RORCO Real' : load_rorco_real,
    'RORCO' : load_rorco,
}