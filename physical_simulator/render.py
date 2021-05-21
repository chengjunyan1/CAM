import bpy,sys,os,time
import numpy as np

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path: sys.path.append(dir)


""" How to use
1. Rotate x by 90d in shader editor
2. Use generated coord
"""


""" 
Settings 
"""

# Base setting
device='GPU'
outputdir='./outputs/'
if not os.path.exists(outputdir): os.makedirs(outputdir)

# Camera settings
ImageWidth = 1920
ImageHeight = 1080
ox = ImageWidth/2
oy = ImageHeight/2
HV=[64,41]

# Light settings
height=9
power=1000
offset={'top':-0.3, 'left':-1.8, 'right':1}

# Sheet settings
bounding=[[0,1],[0,1]] # height,width
strip=5
conform_height=0.035 #0.035
subdiv_level=3
subdiv_quality=2

SCALE=3

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

def set_origin(obj):
    if obj in bpy.data.objects:
        select_obj(obj)
        bpy.ops.object.origin_set()

def move_obj(obj,loc):
    if obj in bpy.data.objects:
        bpy.data.objects[obj].location=np.array(loc)
        
def rotate_obj(obj,deg):
    if obj in bpy.data.objects:
        bpy.data.objects[obj].rotation_euler = np.deg2rad(deg)
        
def create_mesh(name,vertices,edges,faces):
    if name not in bpy.data.objects:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, edges, faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection=bpy.data.collections['Collection']
        collection.objects.link(obj)
        return obj,mesh
    else:
        return bpy.data.objects[name],bpy.data.objects[name].data

def set_camera_FOV(camera_name, HV):
    cam=bpy.data.cameras[camera_name]
    cam.angle_x=np.deg2rad(HV[0])
    cam.angle_y=np.deg2rad(HV[1])


# ---------- SHEET CREATE ---------- #

def scan_surface(scale=11):
    # Scan mold surface 
    # corner_ur=[5.5,3.0939,1]
    # corner_ul=[-5.5,3.0939,0]
    cplane1='Plane' # create calibrating planes
    cplane2='Plane.001'
    if cplane1 not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add() 
        # bpy.ops.transform.resize(value=(10, 10, 0))
        bpy.data.objects[cplane1].location=np.array((-2,-1.5,-1))
    if cplane2 not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add() 
        bpy.data.objects[cplane2].location=np.array((2,1.5,1))

    bpy.data.objects['Camera'].data.type='ORTHO' # set cam to ortho
    bpy.data.objects['Camera'].data.ortho_scale=scale

    bpy.context.scene.use_nodes = True # init node
    tree = bpy.context.scene.node_tree
    for n in tree.nodes: tree.nodes.remove(n)

    rl = tree.nodes.new('CompositorNodeRLayers') # set nodes to output z
    m = tree.nodes.new('CompositorNodeNormalize')   
    v = tree.nodes.new('CompositorNodeViewer')   
    tree.links.new(rl.outputs[2], m.inputs[0])
    tree.links.new(m.outputs[0], v.inputs[0]) 

    bpy.ops.render.render() # render and get z
    pixels = bpy.data.images['Viewer Node'].pixels
    width = bpy.context.scene.render.resolution_x 
    height = bpy.context.scene.render.resolution_y
    z = np.array(pixels).reshape(height,width,4)[:,:,0]
    z*=z!=1 # keep only obj depth 

    bpy.data.objects['Camera'].data.type='PERSP' # recover cam and scene
    delete_obj(cplane1)
    delete_obj(cplane2)
    tree.links.new(rl.outputs[0],v.inputs[0]) # recover nodes
    return z

def scan2coord(i,j,z):
    corner_ur=[5.5*SCALE/11,3.0939*SCALE/11,1]
    # corner_ul=[-5.5,3.0939,0]
    vz=2*(1-z[i,j])
    xw=int(j-ImageWidth/2)
    yw=int(i-ImageHeight/2)
    vx=xw*corner_ur[0]*2/ImageWidth
    vy=yw*corner_ur[1]*2/ImageHeight
    return [vx,vy,vz]

def scan2mesh(z,bounding,strip):
    bounding=np.array(bounding).astype(int)
    vertices=[]
    vmap=-np.ones([ImageWidth//strip,ImageHeight//strip]).astype(int)
    x=bounding[0,1]-1
    vid=0
    while x>=bounding[0,0]: #decrease yw, up to down
        y=bounding[1,0]
        while y<bounding[1,1]: # increase xw, left to right
            if z[x,y]!=0:
                vertices.append(scan2coord(x,y,z))
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
    return vertices,edges,faces,vmap


# ---------- WRINKLE GENERATE ---------- #

def save_state():
    state=[]
    s=bpy.data.objects['Sheet'].data
    for i in s.vertices: state.append(np.array(i.co))
    np.save('./temp/state.npy',state)
    return state

# ---------- OUTPUT ---------- #

def scanning(name,outputdir):
    z=scan_surface(SCALE)
    np.save(os.path.join(outputdir,name+'.npy'),z)

def rendering(name,outputdir):
    tree.links.new(rl.outputs[0],v.inputs[0]) # set nodes to rgb output
    print('Rendering image...')
    bpy.ops.render.render(animation=False)
    img_path = os.path.join(outputdir,name+'.png')
    rendered_image = bpy.data.images['Viewer Node']
    rendered_image.save_render(filepath=img_path)

"""
SETUP 
"""


# Create folders
print('Creating folders...')
if not os.path.exists('./temp'): os.makedirs('./temp')

# Clear scene
print('Clearing scene...')
delete_obj('Cube')
delete_obj('Light')

# Set camera
print('Setting camera...')
move_obj('Camera',(0,0,2))
rotate_obj('Camera',(0,0,0))
set_camera_FOV('Camera',HV)

# Set renderer
print('Setting scene...')
scene = bpy.data.scenes['Scene']
scene.render.engine = 'CYCLES'
scene.render.resolution_x = ImageWidth
scene.render.resolution_y = ImageHeight
scene.render.resolution_percentage = 100
scene.cycles.device=device

# Set light
print('Setting light...')
for loc in ['top','left','right']:
    name='Light_'+loc
    if name not in bpy.data.objects:
        light=bpy.data.lights.new(name=name, type='AREA')
        light_obj=bpy.data.objects.new(name=name, object_data=light)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location=(-1,height*offset[loc],height)
        light.energy=power
        light.shape='RECTANGLE'
        rotate_obj(name,(0,0,0))
        light.size=10
        light.size_y=1


output_dir='/home/acjy777/chengjunyan1/Robotics/Wrinkles/fem/outputs/'
if not os.path.exists(output_dir+'rgb'): os.makedirs(output_dir+'rgb')
outputs=[]
for i in os.listdir(output_dir):
    if i!='Boeing_Sheet_Region1.obj': outputs.append(i)

latest=max(outputs)
name=latest.split('.')[0]
if name not in bpy.data.objects:
    for i in outputs:
        if i.split('.')[0] in bpy.data.objects: delete_obj(i.split('.')[0])
    bpy.ops.import_scene.obj(filepath=output_dir+latest)
    set_origin(name)
    s=bpy.data.objects[name]
    s.modifiers.new('subdiv','SUBSURF') # turn on surface subdivision
    s.modifiers['subdiv'].levels=subdiv_level
    s.modifiers['subdiv'].quality=subdiv_quality
    s.modifiers['subdiv'].show_viewport=False
    m=bpy.data.materials['Material'] # set material
    for i in bpy.data.objects: 
        if i.type=='MESH': i.data.materials.clear()
    s.data.materials.append(m)

# rendering(name,output_dir+'rgb')