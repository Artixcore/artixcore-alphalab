import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor
class ArtixcoreAlphaLabPredictor(Predictor):
 def __init__(s):s.c=s.m=s.u=s.d=s.b=s.r=s.x=None;s.a=[];s.ok=False
 def _p(s,f):
  if not isinstance(f,pd.DataFrame):f=pd.DataFrame(f)
  n=[str(x).lower() if x is not None else'' for x in f.columns.names];u=[len(pd.Index(f.columns.get_level_values(i)).unique()) for i in range(f.columns.nlevels)];fi=next((i for i,x in enumerate(n) if'feature'in x),int(np.argmin(u)));ai=next((i for i,x in enumerate(n) if'asset'in x or'ticker'in x or'symbol'in x),max([i for i in range(f.columns.nlevels) if i!=fi],key=lambda i:u[i]));a=list(dict.fromkeys(f.columns.get_level_values(ai)));q=[]
  for j,z in enumerate(dict.fromkeys(f.columns.get_level_values(fi))):
   c=[x for x in f.columns if x[fi]==z];b=f.loc[:,c].copy();b.columns=[x[ai] for x in c];b=b.reindex(columns=a).apply(pd.to_numeric,errors='coerce');q.append(pd.concat({str(z)+'r':b},axis=1))
   if j<3:q.append(pd.concat({str(z)+'d':b-b.shift(1)},axis=1))
  p=pd.concat(q,axis=1);p.columns.names=['feature','asset'];return p,a
 def _g(s,p):return p.stack(level='asset',future_stack=True).replace([np.inf,-np.inf],np.nan)
 def _q(s,x,y,w,a):
  t=np.sqrt(w);z=x*t[:,None];g=z.T@z;g.flat[::g.shape[0]+1]+=a
  try:return np.linalg.solve(g,z.T@(y*t)).astype('float32')
  except:return(np.linalg.pinv(g)@(z.T@(y*t))).astype('float32')
 def train(s,f,y):
  try:
   p,a=s._p(f);x=s._g(p);t=(y.unstack(-1)if isinstance(y,pd.Series)and isinstance(y.index,pd.MultiIndex)else pd.DataFrame(y,index=p.index)).reindex(index=p.index,columns=a).stack(future_stack=True);x,t=x.align(t,join='inner',axis=0);k=t.notna();x,t=x[k],t[k];s.c=list(x.columns[:35]);x=x[s.c];n=len(x);i=np.arange(n)if n<=80000 else np.unique(np.r_[np.linspace(0,n-48001,32000,dtype=int),np.arange(n-48000,n)]);x=x.iloc[i];t=t.iloc[i];v=x.to_numpy('float32');v[~np.isfinite(v)]=np.nan;s.m=np.nanmedian(v,0);s.m[~np.isfinite(s.m)]=0;k=~np.isfinite(v);v[k]=np.take(s.m,np.where(k)[1]);s.u=v.mean(0);s.d=v.std(0);s.d[s.d<1e-6]=1;v=(v-s.u)/s.d;z=np.nan_to_num(t.to_numpy('float32'));age=(len(i)-1)-np.arange(len(i));w=np.exp(-age/max(1,len(i)*.2));w/=w.mean();s.b=s._q(v,z-z.mean(),w,8);rr=((t.groupby(level=0).rank(pct=True)-.5)*2).to_numpy('float32');s.r=s._q(v,rr-rr.mean(),w,20);s.x=xgb.train({'objective':'reg:squarederror','max_depth':2,'eta':.05,'subsample':.8,'colsample_bytree':.8,'min_child_weight':200,'tree_method':'hist','verbosity':0,'nthread':2,'seed':42},xgb.DMatrix(v,label=z,weight=w),15);s.z=float(z.mean());s.h=float(rr.mean());s.e=float(np.std(z)/max(np.std(rr),1e-6));s.a=a;s.ok=True
  except Exception:s.ok=False
  return s
 def predict(s,f):
  try:
   p,a=s._p(f);x=s._g(p);v=x.reindex(columns=s.c).to_numpy('float32');k=~np.isfinite(v);v[k]=np.take(s.m,np.where(k)[1]);v=(v-s.u)/s.d;o=.72*(s.z+v@s.b)+.24*s.x.predict(xgb.DMatrix(v))+.04*s.e*(s.h+v@s.r);o=pd.Series(np.nan_to_num(o),index=x.index).unstack(-1).reindex(index=f.index,columns=a).fillna(0);return o.sub(o.mean(1),axis=0).astype('float32')
  except Exception:return pd.DataFrame(0.,index=f.index,columns=s.a)
