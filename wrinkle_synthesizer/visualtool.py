import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from math import factorial
from itertools import product
from PIL import Image,ImageDraw
import random,os


ImageWidth = 1920
ImageHeight = 1080
outputdir='./wrinkles_outputs/'
z=np.load(outputdir+'z.npy')
z*=z!=1
bounding=[[0,1],[0.2,0.7]] # height,width
bound= (np.array(bounding)*np.array([[ImageHeight],[ImageWidth]])).astype(int)


""" Create sheet """

corner_ur=[5.5,3.0939,1]
corner_ul=[-5.5,3.0939,0]
bounding=np.array([[0,ImageHeight],[0,ImageWidth]])
strip=10

vertices=[]
vmap=-np.ones([ImageWidth//strip,ImageHeight//strip]).astype(int)
x=bounding[0,1]-1
vid=0
while x>=bounding[0,0]: #decrease yw, up to down
    y=bounding[1,0]
    while y<bounding[1,1]: # increase xw, left to right
        if z[x,y]!=0:
            vz=1-z[x,y]
            xw=int(y-ImageWidth/2)
            yw=int(x-ImageHeight/2)
            vx=xw*corner_ur[0]*2/ImageWidth
            vy=yw*corner_ur[1]*2/ImageHeight
            vertices.append((vx,vy,vz))
            vmap[y//strip,(ImageHeight-x-1)//strip]=vid # build a map of vertices, start from (0,0)
            vid+=1
        y+=strip
    x-=strip    

single_ids=[]
for i in range(ImageWidth//strip): # remove all single points
    for j in range(ImageHeight//strip):
        if vmap[i,j]!=-1: 
            if i>0:
                if j>0:
                    if vmap[i-1,j]!=-1 and vmap[i-1,j-1]!=-1 and vmap[i,j-1]!=-1: continue
                if j<ImageHeight//strip-1:
                    if vmap[i-1,j+1]!=-1 and vmap[i-1,j]!=-1 and vmap[i,j+1]!=-1: continue
            if i<ImageWidth//strip-1:
                if j>0:
                    if vmap[i+1,j]!=-1 and vmap[i,j-1]!=-1 and vmap[i+1,j-1]!=-1: continue
                if j<ImageHeight//strip-1:
                    if vmap[i+1,j+1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j+1]!=-1: continue
            single_ids.append(vmap[i,j])
            vmap[i,j]=-1
vertices = [vertices[i] for i in range(0, len(vertices), 1) if i not in single_ids]
for i in range(ImageWidth//strip): # update vertices and map
    for j in range(ImageHeight//strip):
        if vmap[i,j]!=-1: 
            for s in single_ids:
                if vmap[i,j]>s: vmap[i,j]-=1

cj=ImageHeight//strip-1
ci=ImageWidth//strip-1    
edges=[]
faces=[] # vertices clockwise
for i in range(ImageWidth//strip): # build faces and edges
    for j in range(ImageHeight//strip):
        if j<cj and  i<ci and vmap[i,j]!=-1:
                if vmap[i+1,j+1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j+1]!=-1: 
                    faces.append((vmap[i,j],vmap[i+1,j],vmap[i+1,j+1],vmap[i,j+1]))
                    edges.append((vmap[i,j],vmap[i+1,j]))
                    edges.append((vmap[i,j],vmap[i,j+1]))
                    edges.append((vmap[i+1,j+1],vmap[i+1,j]))
                    edges.append((vmap[i+1,j+1],vmap[i,j+1]))
edges=list(set(edges))    


cmap=-np.ones([ImageWidth//strip,ImageHeight//strip]).astype(int) # corner points
for i in range(ImageWidth//strip): 
    for j in range(ImageHeight//strip):
        if vmap[i,j]!=-1:
            if j<cj:
                    if i<ci and vmap[i+1,j+1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j+1]!=-1: 
                        if (i==0 or vmap[i-1,j]==-1) and (j==0 or vmap[i,j-1]==-1): cmap[i,j]=vmap[i,j]
                    if i>0 and vmap[i-1,j+1]!=-1 and vmap[i-1,j]!=-1 and vmap[i,j+1]!=-1: 
                        if (i==ci or vmap[i+1,j]==-1) and (j==0 or vmap[i,j-1]==-1): cmap[i,j]=vmap[i,j]
            if j>0:
                    if i<ci and vmap[i+1,j-1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j-1]!=-1: 
                        if (i==0 or vmap[i-1,j]==-1) and (j==cj or vmap[i,j+1]==-1): cmap[i,j]=vmap[i,j]
                    if i>0 and vmap[i-1,j-1]!=-1 and vmap[i-1,j]!=-1 and vmap[i,j-1]!=-1: 
                        if (i==ci or vmap[i+1,j]==-1) and (j==cj or vmap[i,j+1]==-1): cmap[i,j]=vmap[i,j]
            if 0<j<cj and 0<i<ci:
                if vmap[i+1,j+1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j+1]!=-1: 
                    if vmap[i-1,j-1]==-1 and vmap[i-1,j]!=-1 and vmap[i,j-1]!=-1: cmap[i,j]=vmap[i,j]
                if vmap[i+1,j-1]!=-1 and vmap[i+1,j]!=-1 and vmap[i,j-1]!=-1: 
                    if vmap[i-1,j+1]==-1 and vmap[i-1,j]!=-1 and vmap[i,j+1]!=-1: cmap[i,j]=vmap[i,j]
                if vmap[i-1,j+1]!=-1 and vmap[i-1,j]!=-1 and vmap[i,j+1]!=-1: 
                    if vmap[i+1,j-1]==-1 and vmap[i+1,j]!=-1 and vmap[i,j-1]!=-1: cmap[i,j]=vmap[i,j]
                if vmap[i-1,j-1]!=-1 and vmap[i-1,j]!=-1 and vmap[i,j-1]!=-1: 
                    if vmap[i+1,j+1]==-1 and vmap[i+1,j]!=-1 and vmap[i,j+1]!=-1: cmap[i,j]=vmap[i,j]

face=[]
stop=0
for i in range(ci+1): # start from upper left corner point, then go counter wise
    for j in range(cj+1):
        if cmap[i,j]!=-1: 
            face.append([i*strip,j*strip])
            curi=i
            curj=j
            stop=1
            break
    if stop: break
cinit=cmap[curi,curj]
move=0 #0 right 1 down 2 left 3 up
while True:  # vertices clockwise
    if move==0:
        for j in range(curj+1,cj+1):
            if cmap[curi,j]!=-1: 
                curj=j
                if curi==0: move=1
                else:
                    found=0
                    for i in range(curi+1,ci+1):
                        if cmap[i,curj]!=-1:
                            if [i,curj] not in face: found=1
                            break
                    move=1 if found else 3
                break
    elif move==1:
        for i in range(curi+1,ci+1):
            if cmap[i,curj]!=-1: 
                curi=i
                if curi==ci: move=2
                else:
                    found=0
                    for j in range(curj+1,cj+1):
                        if cmap[curi,j]!=-1: 
                            if [curi,j] not in face: found=1
                            break
                    move=0 if found else 2
                break
    elif move==2:
        for j in reversed(range(0,curj)):
            if cmap[curi,j]!=-1: 
                curj=j
                if curj==cj: move=3
                else:
                    found=0
                    for i in range(curi+1,ci+1):
                        if cmap[i,curj]!=-1: 
                            if [i,curj] not in face: found=1
                            break
                    move=1 if found else 3
                break
    elif move==3:
        for i in reversed(range(0,curi)):
            if cmap[i,curj]!=-1: 
                curi=i
                if curj==0: move=0
                else:
                    found=0
                    for j in reversed(range(0,curj)):
                        if cmap[curi,j]!=-1: 
                            if [curi,j] not in face: found=1
                            break
                    move=2 if found else 0
                break
    if cmap[curi,curj]==cinit: break
    face.append([curi*strip,curj*strip])
face.append(face[0])
cornerp=np.array(face)



""" Winkles """

def scan2coord(i,j,z):
    corner_ur=[5.5,3.0939,1]
    corner_ul=[-5.5,3.0939,0]
    vz=1-z[int(i),int(j)]
    xw=int(j-ImageWidth/2)
    yw=int(i-ImageHeight/2)
    vx=xw*corner_ur[0]*2/ImageWidth
    vy=yw*corner_ur[1]*2/ImageHeight
    return [vx,vy,vz]

def range2bbox(bound): # bound=[[ymin,ymax],[xmin,xmax]]
    return np.array([[bound[0,0],bound[1,0]],[bound[0,0],bound[1,1]],
                     [bound[0,1],bound[1,1]],[bound[0,1],bound[1,0]],
                     [bound[0,0],bound[1,0]]])

def corner2range(corners): # input 2 corner points
    xmin=min(corners[0][0],corners[1][0])
    xmax=max(corners[0][0],corners[1][0])
    ymin=min(corners[0][1],corners[1][1])
    ymax=max(corners[0][1],corners[1][1])
    return np.array([[ymin,ymax],[xmin,xmax]]).astype(int) # notice that its Y, X

def get_bezier(control_points):
    n = len(control_points) - 1
    def comb(n, k): 
        return factorial(n) // (factorial(k) * factorial(n-k))
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*control_points[i] for i in range(n+1))

def bezier_integral(bcurve,stepsize=1e-4):
    p=bcurve(0)
    l=0
    for i in range(1,int(1/stepsize)+1):
        pn=bcurve(i*stepsize)
        l+=np.sqrt(np.sum(np.power(pn-p,2)))
        p=pn
    return l

def bezier(config):
    bcurve=get_bezier(np.array(config['control_points']))
    point_num=config['point_num']
    if point_num==0:
        curve_len=bezier_integral(bcurve)
        point_num=int(curve_len/config['step_size'])
    return np.array([bcurve(t) for t in np.linspace(0, 1, point_num)]).astype(int)


""" Auto Gen """

# Height depends on: radius (it also depends on 3d view height), num nodes, resample time

def gen_corners(config):
    while True:
        x1=int(random.sample(range(bound[1][0],bound[1][1]),1)[0])
        y1=int(random.sample(range(bound[0][0],bound[0][1]),1)[0])
        if not (bound[1][0]<=x1<bound[1][1] and bound[0][0]<=y1<bound[0][1]): continue
        L=random.sample(range(config['lmin'],config['lmax']),1)[0]
        R=np.deg2rad(random.sample(range(360),1)[0])
        x2=int(x1+L*np.cos(R))
        y2=int(y1+L*np.sin(R))
        if bound[1][0]<=x2<bound[1][1] and bound[0][0]<=y2<bound[0][1]:
            if z[y1,x1]!=0 and z[y2,x2]!=0 and z[y1,x2]!=0 and z[y2,x1]!=0: break
    return [[x1,y1],[x2,y2]]

def increase_order(k,bound): # sample more control points
    control_points=[]
    for i in range(k):
        if bound[1][0]==bound[1][1]: x=bound[1][0]
        else: x=int(random.sample(range(bound[1][0],bound[1][1]),1)[0])
        if bound[0][0]==bound[0][1]: y=bound[0][0]
        else: y=int(random.sample(range(bound[0][0],bound[0][1]),1)[0])
        control_points.append([x,y])
    return control_points
        
def gen_control_points(corners,config):
    k=random.sample(range(config['k']+1),1)[0]
    bounds=corner2range(np.array(corners))
    ncp=increase_order(k, bounds)
    control_points=[corners[0]]+ncp+[corners[1]]
    return control_points

def split_curve(points,decay,config):
    p1=random.choice(points)
    x1,y1=p1[0],p1[1]
    while True:
        L=random.sample(range(int(config['lmin']*decay),int(config['lmax']*decay)),1)[0]
        R=np.deg2rad(random.sample(range(360),1)[0])
        x2=int(x1+L*np.cos(R))
        y2=int(y1+L*np.sin(R))
        if bound[1][0]<=x2<bound[1][1] and bound[0][0]<=y2<bound[0][1]:
            if z[y1,x1]!=0 and z[y2,x2]!=0 and z[y1,x2]!=0 and z[y2,x1]!=0: break
    return [[x1,y1],[x2,y2]]


def draw_bbox(corners,mark='k-'):
    bounds=corner2range(np.array(corners))
    boundp=range2bbox(bounds)
    plt.plot(boundp[:, 1], boundp[:, 0], mark)

def direction(corner): return np.array(corner[1])-np.array(corner[0])
def cosangle(a,b): return np.rad2deg(np.arccos(a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))))
def stdgaussian(x,sigma=1): return np.exp(-x**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def gaussian_smooth(k,bw,sigma=1): # sample from -1 to 1
    smooth_noise=[]
    for i in range(k): smooth_noise.append(stdgaussian(i*2/(k-1)-1,sigma)*bw)
    return np.array(smooth_noise)-np.average(smooth_noise)

""" viz utils """

def merge_bbox(bbox):
    xs,ys=[],[]
    for corner in bbox: 
        xs+=[corner[0][0],corner[1][0]]
        ys+=[corner[0][1],corner[1][1]]
    return [[min(xs),min(ys)],[max(xs),max(ys)]]
    
def draw_bbox_img(img,bbox):
    draw = ImageDraw.Draw(img)
    draw.polygon([(bbox[0][0],ImageHeight-bbox[0][1]),(bbox[0][0],ImageHeight-bbox[1][1]),
                  (bbox[1][0],ImageHeight-bbox[1][1]),(bbox[1][0],ImageHeight-bbox[0][1])], outline=(255,0,0))

def corner_collision(ncornerset,cornersets):
    for cornerset in cornersets:
        for bbox in cornerset:
            for corner in ncornerset:
                bmaxx,bminx=max(bbox[0][0],bbox[1][0]),min(bbox[0][0],bbox[1][0])
                bmaxy,bminy=max(bbox[0][1],bbox[1][1]),min(bbox[0][1],bbox[1][1])
                cmaxx,cminx=max(corner[0][0],corner[1][0]),min(corner[0][0],corner[1][0])
                cmaxy,cminy=max(corner[0][1],corner[1][1]),min(corner[0][1],corner[1][1])
                if bminx<cminx<bmaxx and cmaxy>bminy and cminy<bmaxy: return True
                if bminx<cmaxx<bmaxx and cmaxy>bminy and cminy<bmaxy: return True
                if bminy<cminy<bmaxy and cmaxx>bminx and cminx<bmaxx: return True
                if bminy<cmaxy<bmaxy and cmaxx>bminx and cminx<bmaxx: return True
    return False

def read_labels(path):
    with open(path) as f:
        datas=[i.strip() for i in f.readlines()]
        num=int(datas[0])
        cornerset=[]
        for i in range(1,num+1):
            data=[int(j) for j in datas[i].split(' ')[:4]]
            data[1]=ImageHeight-data[1]
            data[3]=ImageHeight-data[3]
            corner=[data[:2],data[-2:]]
            cornerset.append(corner)
    return cornerset 


""" Scanning viz """

# w=z[:,400]
# w[0]=-3
# w[-1]=1
# plt.plot(-np.array(list(range(1080))),-w)


""" Gen viz """

# gen_config={
#     'lmin':500,
#     'lmax':2000, 
#     'k':4,
#     'split':3,
#     'split_decay':0.5,
#     'split_diver':30,
# }
# lmin=50
# lmax=250

# mask=[[0,650],[ImageWidth-1,600]]
# draw_bbox(mask,'r-')

# corners=gen_corners(gen_config)
# control_points=gen_control_points(corners,gen_config)
# draw_bbox(corners)



# wrinkle_config={
#     'control_points':control_points,
#     'point_num': 0, # 0 means auto point num
#     'step_size':10, # step size for auto point num
# }
# cp = np.array(wrinkle_config['control_points'])
# plt.plot(cp[:, 0], cp[:, 1], 'r.')
    
# bp = bezier(wrinkle_config)

# bps=[bp]
# split=random.sample(range(gen_config['split']+1),1)[0]
# cornerset=[corners]
# decay=1
# for i in range(split):
#     decay*=gen_config['split_decay']
#     root=random.choice(range(len(cornerset)))
#     corner=cornerset[root]
#     direct=direction(corner)
#     bp=bps[root]
#     while True:
#         ncorners=split_curve(bp, decay, gen_config)
#         ndirect=direction(ncorners)
#         absangle=abs(cosangle(direct, ndirect))
#         if gen_config['split_diver']<absangle<180-gen_config['split_diver']: break
#     draw_bbox(ncorners)
#     control_points=gen_control_points(ncorners,gen_config)
#     wrinkle_config['control_points']=control_points
#     cp = np.array(wrinkle_config['control_points'])
#     plt.plot(cp[:, 0], cp[:, 1], 'r.')
#     bp2=bezier(wrinkle_config)
#     bps.append(bp2)
#     cornerset.append(ncorners)
    
# for bp in bps:
#     plt.plot(bp[:, 0], bp[:, 1], 'b-')



""" Results viz """

# file='0'
# save_folder='2021-03-01_15:38:12.369491'

# bbox=np.load(outputdir+'box/'+file+'.npy',allow_pickle=True)
# raw=Image.open(outputdir+'rgb/'+file+'.png')
# for i in bbox:
#     for corner in i: draw_bbox(corner)
# for i in bbox:
#     mbox=merge_bbox(np.array(i))
#     draw_bbox_img(raw,mbox)
# raw.show()

# outputdir=outputdir+save_folder+'/'
# zanchor=np.load(outputdir+'dep/Anchor.npy') # seg data
# z0=np.load(outputdir+'dep/'+file+'.npy')
# d=np.abs(zanchor-z0)*255*4
# d=d[::-1,:]
# im = Image.fromarray(d)
# im = im.convert('L')
# im.show()


""" Save all viz """

save_folder='G1'
overwrite=False
outputdir=outputdir+save_folder+'/'
if os.path.exists(outputdir):
    if not os.path.exists(outputdir+'viz/box/'): os.makedirs(outputdir+'viz/box/')
    if not os.path.exists(outputdir+'viz/seg/'): os.makedirs(outputdir+'viz/seg/')
    filelist=os.listdir(outputdir+'rgb')
    if 'Anchor.png' in filelist: del filelist[filelist.index('Anchor.png')]
    files=[int(f.split('.')[0]) for f in filelist]
    for file in sorted(files):
        file=str(file)
        if overwrite or not os.path.exists(outputdir+'viz/box/'+file+'.png'):
            raw=Image.open(outputdir+'rgb/'+file+'.png')
            bbox=np.load(outputdir+'box/'+file+'.npy',allow_pickle=True)
            for i in bbox:
                mbox=merge_bbox(np.array(i))
                draw_bbox_img(raw,mbox)
            raw.save(outputdir+'viz/box/'+file+'.png')
            
            dz=(np.load(outputdir+'seg/'+file+'.npy')*255*4)[::-1,:]
            im = Image.fromarray(dz).convert('L')
            im.save(outputdir+'viz/seg/'+file+'.png')
            print(file+'.png visualizations saved.')         
else:
    print(outputdir,'not exists.')
    

""" Mask viz """

# save_folder='2021-03-01_11:42:54.226835'
# outputdir=outputdir+save_folder+'/'
# raw=Image.open(outputdir+'rgb/0.png')
# masks=[[[0,630],[ImageWidth-1,700]]]
# for mask in masks: draw_bbox_img(raw,mask)
# raw.show()


""" Draw """

# boundp=np.array(list(product(list(bound[0,:]),list(bound[1,:]))))
# boundp=range2bbox(bound)
# plt.plot(boundp[:, 1], boundp[:, 0], 'g-')
# plt.plot(cornerp[:, 0], cornerp[:, 1], 'y-')
# plt.show()


""" Label viz """

# file='77'
# save_folder='gen_test1'
# outputdir=outputdir+save_folder+'/'
# cornersets=read_labels(outputdir+'labels/'+file+'.txt')
# raw=Image.open(outputdir+'rgb/'+file+'.png')
# for cornerset in cornersets: draw_bbox_img(raw,cornerset)
# raw.show()


""" Smooth viz """

# q=gaussian_smooth(10,1,1)
# plt.plot(list(range(len(q))),q)
