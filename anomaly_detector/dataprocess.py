import os,random,time,json
import numpy as np
from PIL import Image, ImageDraw
from utils import pycococreatortools 
from shapely.ops import unary_union
from shapely import geometry as gs
import mmcv


dbname='G1'
basedir='../simulation/wrinkles_outputs/'
datasetdir='./dataset'
ImageWidth = 1920
ImageHeight = 1080
crop=(343,28,1367,1052) # to 1024x1024

WRINKLE_LABEL=0
image_size=[crop[2]-crop[0],crop[3]-crop[1]]
outputdir=os.path.join(datasetdir,dbname)
datadir=os.path.join(basedir,dbname)
datadirs={}
outputdirs={}
for i in ['rgb','seg']: datadirs[i]=os.path.join(datadir, i)
if not os.path.exists(outputdir): os.makedirs(outputdir)


def seg2polys(segdata,th=5e-3,tol=8):
    d=segdata
    if np.sum(d!=0)/(np.sum(d==0)+np.sum(d!=0))>0.42: return False
    d=(d>th).astype(float)[::-1,:]
    poly=pycococreatortools.binary_mask_to_polygon(d,tol)
    polys=[]
    for p in poly: polys.append(gs.Polygon(np.array(p).reshape([-1,2]).tolist()))
    mpoly=unary_union(polys)
    if type(mpoly)==gs.polygon.Polygon: polys=[mpoly]
    else: polys=list(mpoly)
    poly=[]
    area=[]
    bbox=[]
    for i in polys:
        if i.area==0: continue
        exterior=[]
        x, y = i.exterior.coords.xy
        x,y=np.array(x)-crop[0],np.array(y)-crop[1]
        for j in range(len(x)): exterior+=[x[j],y[j]]
        poly.append(exterior)
        area.append(i.area)
        bbox.append([min(x),min(y),max(x)-min(x),max(y)-min(y)])
    return poly,area,bbox

def create_image_info(image_id, file_name, image_size):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
    }

def create_annotation_info(anno_id, img_id, poly, area, bbox):
    return {
        "id": anno_id,
        "image_id": img_id,
        "category_id": WRINKLE_LABEL, 
        "segmentation": [poly],
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


""" Convert to COCO format """


# print('Dataset will be created on',outputdir,'from',datadir)
# ts=time.time()
# passed=[]
# if os.path.exists(datadir):
#     filelist=os.listdir(datadirs['rgb'])
#     if 'Anchor.png' in filelist: del filelist[filelist.index('Anchor.png')]
#     files=[int(f.split('.')[0]) for f in filelist]
#     img_id=0
#     anno_id=0
#     images=[]
#     annotations=[]
#     for term in sorted(files):
#         if term<400: continue
#         print('Processing:',term)
#         term=str(term)
#         segdata=np.load(os.path.join(datadirs['seg'], term+'.npy'))
#         img=Image.open(os.path.join(datadirs['rgb'],term+'.png'))
#         img=img.crop(crop)
#         res=seg2polys(segdata)
#         if res==False:
#             passed.append(term)
#             continue
#         poly,area,bbox=res
#         img.save(os.path.join(outputdir,term+'.png'))
#         for i in range(len(poly)):
#             annotations.append(create_annotation_info(anno_id,img_id,poly[i],area[i],bbox[i]))
#             anno_id+=1
#         images.append(pycococreatortools.create_image_info(img_id,term+'.png', image_size))
#         img_id+=1
#         coco_format_json = dict(
#                         images=images,
#                         annotations=annotations,
#                         categories=[{'id':WRINKLE_LABEL, 'name': 'wrinkle'}]) 
#         mmcv.dump(coco_format_json, os.path.join(outputdir, dbname+'_labels.json'))
# print('Done! Time:',time.time()-ts,'Cleaned:',passed)


""" Viz """

# term='314'
# segdata=np.load(os.path.join(datadirs['seg'], term+'.npy'))
# d=segdata
# d=(d>5e-3).astype(float)[::-1,:]
# poly=pycococreatortools.binary_mask_to_polygon(d,8)
# polys=[]
# for p in poly: polys.append(gs.Polygon(np.array(p).reshape([-1,2]).tolist()))
# mpoly=unary_union(polys)
# if type(mpoly)==gs.polygon.Polygon: polys=[mpoly]
# else: polys=list(mpoly)
# poly=[]
# area=[]
# bbox=[]
# for i in polys:
#     if i.area==0: continue
#     exterior=[]
#     x, y = i.exterior.coords.xy
#     x,y=np.array(x)-crop[0],np.array(y)-crop[1]
#     for j in range(len(x)): exterior+=[x[j],y[j]]
#     poly.append(exterior)
#     area.append(i.area)
#     bbox.append([min(x),min(y),max(x)-min(x),max(y)-min(y)])

# img=Image.open(os.path.join(datadirs['rgb'],term+'.png'))
# img=img.crop(crop)
# draw = ImageDraw.Draw(img)
# for p in poly:
#     polys=[]
#     for i in range(int(len(p)/2)): polys.append((p[i*2],p[i*2+1]))
#     draw.polygon(polys, outline=(255,0,0))
# img.show()



""" Check """

# term=235
# split='train'
# img_dir=os.path.join(outputdir,split)
# with open(os.path.join(outputdir,split+'.json')) as f: label=json.load(f)

# img_ids=[i['id'] for i in label['images']]
# anno_ids=[i['id'] for i in label['annotations']]

# imginfo=label['images'][term]
# img=Image.open(os.path.join(img_dir,imginfo['file_name']))

# draw = ImageDraw.Draw(img)
# for anno in label['annotations']:
#     if anno['image_id']==imginfo['id']:
#         i = anno['bbox']
#         draw.polygon([(i[0],i[1]),(i[0]+i[2],i[1]),
#                       (i[0]+i[2],i[1]+i[3]),(i[0],i[1]+i[3])], outline=(255,0,0))
# img.show() # result: need use image coords in data


""" Test """

# with open('./labels_test.json') as f: test=json.load(f)
# term=1

# image=test['images'][term]
# annos=test['annotations']
# anno=[]
# for i in annos: 
#     if i['image_id']==image['id']: anno.append(i)
# bbox=[i['bbox'] for i in anno]
# corner=[]
# for i in bbox:
#     corner.append([[i[0],i[1]],[i[0]+i[2],i[1]+i[3]]])

# img=Image.open(os.path.join(datadirs['rgb'],image['file_name']))
# draw = ImageDraw.Draw(img)
# for i in corner:
#     draw.polygon([(i[0][0],i[0][1]),(i[0][0],i[1][1]),
#                   (i[1][0],i[1][1]),(i[1][0],i[0][1])], outline=(255,0,0))
# img.show() # result: need use image coords in data


""" Process real """

# dbname='R_Pics'
# rawpath='./dataset/Pics/Pics'
# dbdir=os.path.join(datasetdir,dbname)
# files=os.listdir(rawpath)
# if not os.path.exists(dbdir): os.makedirs(dbdir)
# # crop=[2160,80,3330,1040]
# crop=[210,50,1020,720]
# for i in files:
#     print('Processing',i)
#     img=Image.open(os.path.join(rawpath,i))
#     img=img.crop(crop)
#     img.save(os.path.join(dbdir,i))


# img=Image.open(os.path.join(rawpath,'4_Color.png'))
# img=img.crop(crop)
# img.show()


""" Split Labels """


labelfile='G1/G1_labels'

labels=mmcv.load(os.path.join(datasetdir, labelfile+'.json'))

split=['train','test','val']
train_split=0.5
test_split=0.5
pass_ids=[]

splits={}
files=[]
for i in labels['images']:
    if i['id'] not in pass_ids: files.append(i) 
random.shuffle(files)
splits['train']=files[:int(len(files)*train_split)]
splits['test']=files[int(len(files)*train_split):int(len(files)*(train_split+test_split))]
splits['val']=files[int(len(files)*(train_split+test_split))::]

newlabels={}
for i in split: 
    if splits[i]==[]: continue
    newlabels[i]={}
    # newlabels[i]['info']=labels['info'] 
    newlabels[i]['categories']=labels['categories']
    newlabels[i]['images']=splits[i]
    newlabels[i]['annotations']=[]
    for img in splits[i]:
        for j in labels['annotations']:
            if j['image_id']==img['id']: newlabels[i]['annotations'].append(j)
    mmcv.dump(newlabels[i],os.path.join(datasetdir, labelfile+'_'+i+'.json'))

   

""" Move test imgs """

# dbname='R_Pics'
# labelfile='./'+dbname+'/Label_R_track_test'

# labels=mmcv.load(os.path.join(datasetdir, labelfile+'.json'))
# img_dir=os.path.join(datasetdir, dbname)
# img_test_dir=os.path.join(img_dir, 'test_track')
# if not os.path.exists(img_test_dir): os.makedirs(img_test_dir)
# for i in labels['images']:
#     img=Image.open(os.path.join(img_dir,i['file_name']))
#     img.save(os.path.join(img_test_dir,i['file_name']))

             
""" Make a pseudo labelfile """ 

# name='test'
# path=os.path.join(datasetdir, 'R_Pics/test')
# if os.path.exists(path):
#     filelist=os.listdir(path)
#     img_id=0
#     anno_id=0
#     images=[]
#     annotations=[]
#     for term in sorted(os.listdir(path)):
#         if not term.split('.')[-1]=='png': continue
#         print('processing',term)
#         term=str(term)
#         annotations.append(create_annotation_info(anno_id,img_id,[[]],0,[]))
#         anno_id+=1
#         images.append(pycococreatortools.create_image_info(img_id,term, image_size))
#         img_id+=1
#         coco_format_json = dict(
#                         images=images,
#                         annotations=annotations,
#                         categories=[{'id':WRINKLE_LABEL, 'name': 'wrinkle'}]) 
#         mmcv.dump(coco_format_json, os.path.join(path, name+'_pseudo_labels.json'))

    

""" Simple crop """

# datadir='./dataset/S_1_good'
# savedir=os.path.join(datadir, 'crop')
# if not os.path.exists(savedir): os.makedirs(savedir)
# terms=[]
# for i in os.listdir(datadir):
#     if i.endswith('.png'): terms.append(i)
# crop=(343+384,28,1367+32,1052)
# for i in terms:
#     img=Image.open(os.path.join(datadir,i))
#     img=img.crop(crop)
#     img.save(os.path.join(savedir,i))
    
    
    
    

    

