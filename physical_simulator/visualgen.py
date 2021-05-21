import os,sys
import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


""" Global config """
prec=1.1e-2 # control accu of fp
HOLD=0
SCALE=5
DIS_PERT=0.03

# initialization
if len(sys.argv)>3:
    ifile,ofile,basedir=sys.argv[1],sys.argv[2],sys.argv[3]
else:
    ifile='simulator_constraint.cpp'
    # ofile='/home/acjy777/catkin_ws/src/composite_simulator/src/simulator_constraint_test.cpp'
    ofile='simulator_constraint_test.cpp'
    basedir='.'

vs=np.load(basedir+'/vs.npy')
vs=vs[:,[0,2,1]] # to xyz
vs[:,0]/=max(vs[:,0]) # norm to [0,1]
vs[:,1]/=max(vs[:,1])

fig = plt.figure()
ax1 = plt.axes(projection='3d')
glax={0:0,1:2,2:1}
    
def preparefps(fps):
    global ss,ps
    ss,ps={},{}
    for i in fps:
        ss[i]=draw3fp(fps[i])
        ps[i]=meanpoint(fps[i])
    ax1.plot3D([0,0,1,1,0],[0,1,1,0,0],[0,0,0,0,0])
    ax1.plot3D([1],[1],[0.5],'black')
    ax1.plot3D([0],[0],[-0.5],'black')
    reportpoints(ss)
    
def eudist(p,vs=vs): return np.sqrt(np.sum(np.power(vs-p,2),1))
def getpoint(p,vs=vs,prec=prec): return np.where(eudist(np.array(p),vs[:,:2])<=prec)[0].tolist()
def draw2vs(vs=vs): plt.plot(vs[:,0],vs[:,1])
def draw2fp(fp,vs=vs):
    select=getpoint(fp,vs)
    plt.plot(vs[select,0],vs[select,1],'r*')
def draw_line(line,c='orange'): ax1.plot3D([line[0][0],line[1][0]],[line[0][1],line[1][1]],[0,0],c)
    
def draw3fp(p,ax=ax1): 
    fp,c=p
    select=getpoint(fp,vs)
    p=np.mean(vs[select,:],0)
    ax.plot3D([p[0]],[p[1]],[p[2]], c+'*')
    return [select,c]

def draw_constraints(constraints):
    for constraint in constraints: draw_line(constraint,'r--')

def line_constraint(constraint):
    start,end=np.array(constraint[0]),np.array(constraint[1])
    selects=[]
    pts=[start+t*(end-start) for t in np.linspace(0, 1, 1000)]
    for pt in pts: selects+=getpoint(pt)
    return list(set(selects))

def line_constrains(constraints):
    selects=[]
    for i in constraints: selects.append(line_constraint(i))
    return selects

def meanpoint(p):
    fp,c=p
    select=getpoint(fp,vs)
    return [np.mean(vs[select,:],0),c]

def draw3move(ps,axis,s,t,c='r',ax=ax1):
    steps=np.linspace(0, s*t,100)
    mvs=np.repeat(np.expand_dims(ps,0),100,0)
    mvs[:,axis]+=steps
    ax.plot3D(mvs[:,0],mvs[:,1],mvs[:,2], c)
    return mvs[-1,:]

def normvec(vec): return np.array(vec)/np.sqrt(np.sum(np.power(vec,2)))

def moveline(p,time,vec,ss,v,c):
    tnow,end=0,0
    vec=normvec(vec)
    assert np.sum(np.abs(vec))!=0
    while True:
        for i in range(3):
            if vec[i]!=0: 
                step=time-tnow if time-tnow<ss else ss
                p=draw3move(p,i,v*vec[i],step,c)
            tnow+=ss
            if tnow>=time:
                end=1
                break
        if end: break
    return p

def checkmoves(moves):
    check=0
    for move in moves:
        assert move[0][0]>=check, 'time should not intersect'
        check=move[0][1]

def draw3moves(moves,ps):
    checkmoves(moves)
    for move in moves:
        time=move[0][1]-move[0][0]
        ss=move[1]
        for line in move[2]:
            if len(line)!=1:
                p,v,vec=line
                ps[p][0]=moveline(ps[p][0],time,vec,ss,v/SCALE,ps[p][1])

def reportpoints(ss): 
    print('VERTICES IDS OF FIXED POINTS:')
    for i in ss: print(i,ss[i])

def perturbation(v):
    return v*(1-DIS_PERT+2*DIS_PERT*np.random.rand())

def getlineslides(moves,fps,cps):
    slides=[]
    for move in moves:
        window,sec,lines=move
        ts=window[0]
        dur=window[1]-ts
        assert dur%sec==0, 'step size should be dividable by duration'
        snow=0
        while True:
            stop=0
            for ax in [0,1,2]:
                if snow*sec>=dur:
                    stop=1
                    break
                slide=[ts+snow*sec,ts+(snow+1)*sec]
                actions=[]
                moved=[]
                for line in lines:
                    p,v,vec=line
                    vec=normvec(vec)
                    if vec[ax]!=0: 
                        vax=perturbation(v*vec[ax])
                        actions.append([p,glax[ax],vax])
                    else: actions.append([p,0,HOLD])
                    moved.append(p)
                for i in fps:
                    if i not in moved: actions.append([i,0,HOLD])
                for i in range(len(cps)): actions.append(['constraint'+str(i),0,HOLD])
                slides.append([slide,actions])
                snow+=1
            if stop: break
    return slides

def genmovescode(slides,fps,cps):
    codes={}
    s=''
    for slide in slides:
        window,actions=slide
        newline='if(timeStepCount<'+str(window[1])+' && timeStepCount>='+str(window[0])+'){\n'
        newline='\telse '+newline if s!='' else '\t'+newline
        for action in actions:
            p,axis,v=action
            newline+='\t\tsheet->MoveSurfaceTo('+p+','+str(axis)+','+str(v)+'*t);\n'
        s+=newline+'\t}\n'
    lastline='\telse if(timeStepCount>='+str(window[1])+'){\n'
    for fp in fps:
        lastline+='\t\tsheet->MoveSurfaceTo('+fp+','+str(0)+','+str(HOLD)+'*t);\n'
    for i in range(len(cps)): 
        lastline+='\t\tsheet->MoveSurfaceTo(constraint'+str(i)+','+str(0)+','+str(HOLD)+'*t);\n'
    s+=lastline+'\t}\n'
    codes['moves']=s
    codes['fpdefine']='vector<int> '
    codes['fpssetting']=''
    count=0
    for i in fps: 
        codes['fpdefine']+=i
        if count!=len(fps)-1: codes['fpdefine']+=','
        fpset='\t'+i+' = std::vector<int>{'
        fpset+=str(fps[i][0])[1:-1]
        fpset+='};\n'
        codes['fpssetting']+=fpset
        count+=1
    for i in range(len(cps)): 
        codes['fpdefine']+=',constraint'+str(i)
        fpset='\tconstraint'+str(i)+' = std::vector<int>{'
        fpset+=str(cps[i])[1:-1]
        fpset+='};\n'
        codes['fpssetting']+=fpset
    codes['fpdefine']+=';\n'
    return codes

def findinsertmark(mark,source):
    insert=-1
    for i in range(len(source)): 
        if mark in source[i]: 
            insert=i+1
            break
    assert insert!=-1, mark+' not found'
    return insert

def insertcodes(codes,ifile,ofile):
    with open(ifile,'r') as f: source=f.readlines()
    insert=findinsertmark('INSERT_GEN_FPS_DEFINE_MARK',source)
    source.insert(insert,codes['fpdefine'])
    insert=findinsertmark('INSERT_GEN_MOVES_CODE_MARK',source)
    source.insert(insert,codes['moves'])
    insert=findinsertmark('INSERT_GEN_FPS_SETTING_MARK',source)
    source.insert(insert,codes['fpssetting'])
    with open(ofile,'w') as f: f.writelines(source)


""" shortcuts """
def easygen(fps,moves,constraints,ifile,ofile):
    preparefps(fps)
    draw3moves(moves,ps)     
    draw_constraints(constraints)
    cps=line_constrains(constraints)
    slides=getlineslides(moves,fps,cps)
    s=genmovescode(slides,ss,cps)
    insertcodes(s,ifile,ofile)
    print('Animation code generated.')
    
def hang(d,v,th,r,pl='fp1',pr='fp2'):
    h=r*np.sin(np.deg2rad(d))
    l=r*np.cos(np.deg2rad(d))-r
    return [[0,th],1,[[pl,v,[l,0,h]],[pr,v,[l,0,h]]]]
    

if __name__=='__main__':
    
    fps=dict(
        fp1=[[0.995,0.005],'r'],
        fp2=[[0.995,0.995],'b'],
        # fp3=[[0.005,0.995],'g'],
        # fp4=[[0.005,0.005],'y'],
    )
    v=0.5
    ct=0.1
    d=30
    th=3
    moves=[
        hang(d,v,th,1-ct),
        [[th,th+10],1,[['fp1',v,[-0.2,0,1]],['fp2',v,[-0.2,0,1]]]],
        # [[th+5,th+10],1,[['fp1',v,[-2,1,0]]]],
    ]
    constraints=[
        [[ct,0],[ct,1.0]],
        [[ct,1.0],[0,1.0]],
        [[0,0],[0,1.0]],
        [[0,0],[ct,0]],
    ]

    easygen(fps,moves,constraints,ifile,ofile)

    

    
    # def get_bezier(control_points):
    #     n = len(control_points) - 1
    #     def comb(n, k): 
    #         return factorial(n) // (factorial(k) * factorial(n-k))
    #     return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*control_points[i] for i in range(n+1))
    
    # def bezier_integral(bcurve,stepsize=1e-4):
    #     p=bcurve(0)
    #     l=0
    #     for i in range(1,int(1/stepsize)+1):
    #         pn=bcurve(i*stepsize)
    #         l+=np.sqrt(np.sum(np.power(pn-p,2)))
    #         p=pn
    #     return l
    
    # def bezier(cps,pnum):
    #     bcurve=get_bezier(np.array(cps))
    #     return np.array([bcurve(t) for t in np.linspace(0, 1, pnum)])

    # cps=[[0,0,0],[0,0.1,0.1],[0.1,0.2,0.2]]
    # r=bezier(cps,40)
