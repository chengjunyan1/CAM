import bpy,sys,os,time,random,datetime
import numpy as np
from math import factorial

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path: sys.path.append(dir)


""" How to use
1. Click Sheet
2. Set 3D view (for this setting, mouse wheel up at least 5 times)
3. Run the script
"""


""" 
Settings 
"""

# Base setting
outputdir='./wrinkles_outputs/'
save_foler=None
save_dep=False
debug_mode=False

# Camera settings
ImageWidth = 1920
ImageHeight = 1080

# Sampling settings
sampling_num=100
masks=[[[0,630],[ImageWidth-1,700]]] # regions masked, assign corner points


""" inits """

z=np.load(os.path.join(outputdir,'z.npy'))
bounding=np.load(os.path.join(outputdir,'bounding.npy'))
outputdirs={}
subdirs=['rgb','seg','dep','box','viz'] 
samp_log={
    'accomplished': 0
}
if not debug_mode:
    if save_foler is None: save_foler=str(datetime.datetime.now()).replace(' ','_')
else: save_foler='debug'
outputdir=os.path.join(outputdir,save_foler)
print('\nResults will be saved in',outputdir)
if not debug_mode:
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        with open(os.path.join(outputdir,'log.txt'),'w') as f: f.write(str(samp_log))
        for i in subdirs: 
            outputdirs[i]=os.path.join(outputdir,i)
            os.makedirs(outputdirs[i])
    else:
        for i in subdirs:  outputdirs[i]=os.path.join(outputdir,i)
        found_ckpt=1
        print('Found checkpoint.')
        with open(os.path.join(outputdir,'log.txt'),'r') as f: samp_log=eval(f.read().strip())

ox = ImageWidth/2
oy = ImageHeight/2
bound= (bounding*np.array([[ImageHeight],[ImageWidth]])).astype(int)

bpy.context.scene.use_nodes = True # init node
tree = bpy.context.scene.node_tree
for n in tree.nodes: tree.nodes.remove(n)

rl = tree.nodes.new('CompositorNodeRLayers') # add nodes
m = tree.nodes.new('CompositorNodeNormalize')   
v = tree.nodes.new('CompositorNodeViewer')  


""" 
Utils
"""

# ---------- GENERAL TOOLS ---------- #

def select_obj(obj):
    if obj in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[obj].select_set(True)

def delete_obj(obj):
    if obj in bpy.data.objects:
        select_obj(obj)
        bpy.ops.object.delete() 

def direction(corner): return np.array(corner[1])-np.array(corner[0])
def cosangle(a,b): return np.rad2deg(np.arccos(a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))))
def stdgaussian(x,sigma=1): return np.exp(-x**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def gaussian_smooth(k,bw,sigma=1): # sample from -1 to 1
    smooth_noise=[]
    for i in range(k): smooth_noise.append(stdgaussian(i*2/(k-1)-1,sigma)*bw)
    return np.array(smooth_noise)-np.average(smooth_noise)

# ---------- SHEET CREATE ---------- #

def scan2coord(i,j,z):
    corner_ur=[5.5,3.0939,1]
    corner_ul=[-5.5,3.0939,0]
    vz=1-z[int(i),int(j)]
    xw=int(j-ImageWidth/2)
    yw=int(i-ImageHeight/2)
    vx=xw*corner_ur[0]*2/ImageWidth
    vy=yw*corner_ur[1]*2/ImageHeight
    return [vx,vy,vz]


# ---------- SCULPT ---------- #

def context_override():
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {'window': window, 'screen': screen, 'area': area, 'region': region, 'scene': bpy.context.scene} 

def sculpt(coordinates, config):
    bpy.data.scenes['Scene'].tool_settings.sculpt.use_symmetry_x=False
    bpy.ops.paint.brush_select(sculpt_tool='CREASE', toggle=False)
    bpy.data.brushes["Crease"].direction = 'ADD'
    bpy.data.scenes['Scene'].tool_settings.unified_paint_settings.use_locked_size='VIEW' # 'SCENE' lock RADIUS, buggy
    bpy.data.brushes['Crease'].use_scene_spacing='VIEW' # 'SCENE' lock spacing

    # Advanced settings
    bpy.data.brushes['Crease'].curve_preset=config['curve_preset']
    bpy.data.brushes['Crease'].auto_smooth_factor=config['auto_smooth_factor']
    bpy.data.brushes['Crease'].hardness=config['hardness']
    bpy.data.brushes['Crease'].normal_radius_factor=config['normal_radius_factor']
    bpy.data.brushes['Crease'].jitter=config['jitter']
    bpy.data.brushes['Crease'].spacing=config['spacing']
    bpy.data.meshes['Sheet'].remesh_voxel_size=config['remesh_voxel_size']
    bpy.data.meshes['Sheet'].remesh_voxel_adaptivity=config['remesh_voxel_adaptivity']
    bpy.data.meshes['Sheet'].use_remesh_smooth_normals=config['use_remesh_smooth_normals']

    strength_smooth=gaussian_smooth(len(coordinates),config['strength_bw'],config['strength_sigma'])
    pinch_smooth=gaussian_smooth(len(coordinates),config['pinch_bw'],config['pinch_sigma'])
    radius_smooth=gaussian_smooth(len(coordinates),config['radius_bw'],config['radius_sigma'])
    for i, coordinate in enumerate(coordinates):
        bpy.data.brushes["Crease"].strength = config['strength']+strength_smooth[i]
        bpy.data.brushes["Crease"].crease_pinch_factor = config['pinch']+pinch_smooth[i]
        bpy.data.scenes['Scene'].tool_settings.unified_paint_settings.unprojected_radius=config['radius']+radius_smooth[i]
        stroke = {
            "name": "stroke",
            "mouse": (0,0),
            "pen_flip" : True,
            "is_start": True if i==0 else False,
            "location": coordinate,
            "size": 50,
            "pressure": 1,
            "time": float(i)
        }
        bpy.ops.sculpt.brush_stroke(context_override(), stroke=[stroke])


# ---------- WRINKLE GENERATE ---------- #

def restore_state(state=None):
    s=bpy.data.objects['Sheet'].data
    if state==None: state=np.load('./temp/state.npy')
    for i in range(len(state)): s.vertices[i].co=state[i]

def point2coord(points,z):
    coords=[]
    for i in list(points): coords.append(scan2coord(i[0],i[1],z))
    return coords

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
    curve_len=bezier_integral(bcurve)
    if config['auto_point_num']: 
        point_num=int(curve_len/config['step_size'])
    else:  point_num=config['point_num']
    coords=np.array([bcurve(t) for t in np.linspace(0, 1, point_num)]).astype(int)
    coords+=np.random.randint(0,config['rand_noise']+1,coords.shape)
    return coords,curve_len

def corner2range(corners): # input 2 corner points
    xmin=min(corners[0][0],corners[1][0])
    xmax=max(corners[0][0],corners[1][0])
    ymin=min(corners[0][1],corners[1][1])
    ymax=max(corners[0][1],corners[1][1])
    return np.array([[ymin,ymax],[xmin,xmax]]).astype(int) # notice that its Y, X

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
    return [[x1,y1],[x2,y2]],L

def bbox_collision(ncornerset,cornersets):
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

def increase_order(k,bound): # sample more control points
    control_points=[]
    for i in range(k):
        if bound[1][0]==bound[1][1]: x=bound[1][0]
        else: x=int(random.sample(range(bound[1][0],bound[1][1]),1)[0])
        if bound[0][0]==bound[0][1]: y=bound[0][0]
        else: y=int(random.sample(range(bound[0][0],bound[0][1]),1)[0])
        control_points.append([x,y])
    return control_points
        
def gen_control_points(config,corners):
    k=random.sample(range(config['k']+1),1)[0]
    bounds=corner2range(np.array(corners))
    ncp=increase_order(k, bounds)
    control_points=[corners[0]]+ncp+[corners[1]]
    return control_points

def split_curve(points,decay,config):
    p1=random.choice(points)
    x1,y1=p1[0],p1[1]
    while True:
        L=random.sample(range(int(decay*config['lmin']),int(decay*config['lmax'])),1)[0]
        R=np.deg2rad(random.sample(range(360),1)[0])
        x2=int(x1+L*np.cos(R))
        y2=int(y1+L*np.sin(R))
        if bound[1][0]<=x2<bound[1][1] and bound[0][0]<=y2<bound[0][1]:
            if z[y1,x1]!=0 and z[y2,x2]!=0 and z[y1,x2]!=0 and z[y2,x1]!=0: break
    return [[x1,y1],[x2,y2]],L

def sampling(config,z):
    Ls=[]
    if config['auto_control_points']:
        corners,L=gen_corners(config)
        control_points=gen_control_points(config,corners)
        config['control_points']=control_points
    else: 
        control_points=config['control_points'] # if not auto, you must assign by yourself
        corners=[control_points[0],control_points[-1]]
        L=np.sqrt(np.sum(np.power(np.array(control_points[0])-np.array(control_points[-1]),2)))
    points,BL=bezier(config)
    Ls.append(L)

    coordset=[point2coord(points[:,::-1],z)]
    pointset=[points]
    cornerset=[corners]
    split=random.sample(range(config['split']+1),1)[0]
    decay=1
    for i in range(split):
        decay*=config['split_decay']
        root=random.choice(range(len(cornerset)))
        corner=cornerset[root]
        direct=direction(corner)
        points=pointset[root]
        while True:
            ncorners,L=split_curve(points, decay, config)
            ndirect=direction(ncorners)
            absangle=abs(cosangle(direct, ndirect))
            if config['split_diver']<absangle<180-config['split_diver']: break
        control_points=gen_control_points(config,ncorners)
        config['control_points']=control_points
        npoints,BL=bezier(config)
        pointset.append(npoints)
        cornerset.append(ncorners)
        coordset.append(point2coord(npoints[:,::-1],z))
        Ls.append(L)
    return coordset,Ls,cornerset


# ---------- OUTPUT ---------- #

def rendering(name,outputdirs):
    tree.links.new(rl.outputs[0],v.inputs[0]) # set nodes to rgb output
    print('Rendering image...')
    bpy.ops.render.render(animation=False)
    img_path = os.path.join(outputdirs['rgb'],name+'.png')
    rendered_image = bpy.data.images['Viewer Node']
    rendered_image.save_render(filepath=img_path)

    tree.links.new(rl.outputs[2], m.inputs[0]) # set nodes to dep output
    tree.links.new(m.outputs[0], v.inputs[0])
    pixels = bpy.data.images['Viewer Node'].pixels
    width = bpy.context.scene.render.resolution_x 
    height = bpy.context.scene.render.resolution_y
    z = np.array(pixels).reshape(height,width,4)[:,:,0]
    if save_dep: np.save(os.path.join(outputdirs['dep'],name+'.npy'),z)
    return z


"""
MAIN 
"""


""" Sampling """

print('Preparing... Please make sure 3D view correctly adjusted.')
tbegin=time.time()
if not debug_mode:
    zanchor=rendering('Anchor',outputdirs) 
    print('Anchor saved.')
select_obj('Sheet')
bpy.ops.object.mode_set(mode='SCULPT')
samp_begin=samp_log['accomplished']
print('Synthesizing start.\n')
if debug_mode: sampling_num=1
for samp in range(samp_begin,sampling_num):
    tstart=time.time()
    config={
        'name':str(samp),
        'auto_control_points':True, #  auto generate control points
        'auto_point_num': True, #  auto adapt point num
        'rand_noise': 1, # rand int noise magnitute on coord, 0 means turn off, simpler than jitter
        # Gen configs
        'step_size':10, # step size for auto point num
        'lmin':50,
        'lmax':125, 
        'k':2, # max bezier curve order
        'split':2, # max curve split num
        'split_decay':0.5, # split decay rate (L gradually smaller)
        'split_diver':30, # min divergence between splits
        'max_samples':2, # max number of samples generated
        # Sculpt configs
        'strength_min': 0.4, # smooth
        'strength_max': 0.6,
        'pinch_min': 0.5, # random  
        'pinch_max': 0.8,
        'radius_min': 0.05, # random
        'radius_max': 0.15,
        # Smooth options
        'strength_bw': 0.1, # gaussian smooth bandwith, 0 means turn off
        'strength_sigma': 0.5, # gaussian smooth sigma
        'pinch_bw': 0.1, 
        'pinch_sigma': 1, 
        'radius_bw': 0.01, 
        'radius_sigma': 1, 
        # Advanced options
        'curve_preset': 'SMOOTH',
        'auto_smooth_factor': 0,
        'hardness': 0,
        'normal_radius_factor': 0.5,
        'jitter': 0,
        'spacing': 10,
        'remesh_voxel_size': 0.1,
        'remesh_voxel_adaptivity': 0,
        'use_remesh_smooth_normals': False,
    }
    samp_num=random.sample(range(config['max_samples']),1)[0]+1
    cornersets=[]
    for s in range(samp_num):
        while True:
            coordset,Ls,cornerset=sampling(config,z)
            check=1
            if not (bbox_collision(cornerset,cornersets) or bbox_collision(cornerset,[masks])): 
                cornersets.append(cornerset)
                for i in range(len(coordset)): 
                    coords=coordset[i]
                    if len(coords)<=1: 
                        check=0
                        break
                    config['strength']=(config['strength_min']+(config['strength_max']-config['strength_min'])
                                                            *(Ls[i]-config['lmin'])/(config['lmax']-config['lmin']))
                    config['pinch']=random.random()*(config['pinch_max']-config['pinch_min'])+config['pinch_min']
                    config['radius']=random.random()*(config['radius_max']-config['radius_min'])+config['radius_min']
                    sculpt(coords,config)
                if check: break
    if not debug_mode:
        np.save(os.path.join(outputdirs['box'],config['name']+'.npy'),np.array(cornersets))
        zi=rendering(config['name'], outputdirs)
        np.save(os.path.join(outputdirs['seg'],config['name']+'.npy'),np.abs(zanchor-zi))
        samp_log['accomplished']+=1
        with open(os.path.join(outputdir,'log.txt'),'w') as f: f.write(str(samp_log))
        print('Synthesized:',str(samp+1)+'/'+str(sampling_num),'\nTime:',time.time()-tstart,'\n')
    restore_state()
bpy.ops.object.mode_set(mode='OBJECT')
print('\nDone!',(sampling_num),'samples synthezed. \nResults saved in',outputdir,'\nTotal time:',time.time()-tbegin)

""" TEST """
# rendering('test', outputdirs['rgb'])
