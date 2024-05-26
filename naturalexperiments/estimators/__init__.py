from .regression_discontinuity import compute_regression_discontinuity
from .propensity_stratification import compute_propensity_stratification
from .direct_difference import compute_direct_difference
from .learned_direct import compute_learned_direct
from .horvitz_thompson import compute_horvitz_thompson
from .doubly_robust import compute_doubly_robust
from .tmle import compute_TMLE
from .direct_prediction import compute_direct_prediction
from .off_policy import compute_off_policy
from .catenet import wrap_catenet
from .double_double import compute_double_double

methods = {
    'Regression Discontinuity' : compute_regression_discontinuity, 
    'Propensity Stratification' : compute_propensity_stratification,
    'Direct Difference' : compute_direct_difference, 
    'Adjusted Direct' : compute_learned_direct,
    'Horvitz-Thompson' : compute_horvitz_thompson,
    'Doubly Robust' : compute_doubly_robust,
    'TMLE' : compute_TMLE,
    'Off-policy' : compute_off_policy,
    'Double-Double' : compute_double_double,
    'Direct Prediction' : compute_direct_prediction,
    'SNet' : wrap_catenet('SNet'),
    'FlexTENet' : wrap_catenet('FlexTENet'),
    'OffsetNet' : wrap_catenet('OffsetNet'),
    'TNet' : wrap_catenet('TNet'),
    'TARNet' : wrap_catenet('TARNet'),
    'DragonNet' : wrap_catenet('DragonNet'),
    'SNet3' : wrap_catenet('SNet3'),
    'DRNet' : wrap_catenet('DRNet'),
    'RANet' : wrap_catenet('RANet'),
    'PWNet' : wrap_catenet('PWNet'),
    'RNet' : wrap_catenet('RNet'),
    'XNet' : wrap_catenet('XNet'),
}