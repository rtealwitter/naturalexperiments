from catenets.models.jax import SNet, FlexTENet, OffsetNet, TNet, TARNet, DragonNet, SNet3, DRNet, RANet, PWNet, RNet, XNet

catenet_models = {'SNet' : SNet, 'FlexTENet' : FlexTENet, 'OffsetNet' : OffsetNet, 'TNet' : TNet, 'TARNet' : TARNet, 'DragonNet' : DragonNet, 'SNet3' : SNet3, 'DRNet' : DRNet, 'RANet' : RANet, 'PWNet' : PWNet, 'RNet' : RNet, 'XNet' : XNet}

def wrap_catenet(model_name):
    def get_catenet_estimate(X, y, z, p, train_fn):
        y = y['y0']*(1-z) + y['y1'] * z
        y = y.values
        w = z
        t = catenet_models[model_name]()
        t.fit(X, y, w)
        cate_pred = t.predict(X)
        return float(cate_pred.mean())
    return get_catenet_estimate