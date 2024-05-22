from .acic import load_acic
from .ihdp import load_ihdp
from .jobs import load_jobs
from .news import load_news
from .rorco import load_rorco
from .twins import load_twins

dataloaders = {
    'ACIC' : load_acic,
    'IHDP' : load_ihdp,
    'JOBS' : load_jobs,
    'NEWS' : load_news,
    'RORCO' : load_rorco,
    'TWINS' : load_twins,
}