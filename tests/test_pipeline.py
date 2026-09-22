import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import dump
from data import CATEGORIES, DetectionDataset, convert
from metrics import evaluate_predictions

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new('RGB', (200,100)).save(self.root/'a.jpg')
        Image.new('RGB', (200,100)).save(self.root/'b.jpg')
        self.labels = [dict(name='a.jpg',labels=[dict(category='person',box2d=dict(x1=20,y1=10,x2=60,y2=30)),
                       dict(category='bike',box2d=dict(x1=100,y1=40,x2=150,y2=80))]),dict(name='b.jpg',labels=[])]
        dump(self.root/'labels.json', self.labels)

    def tearDown(self):
        self.tmp.cleanup()

    def converted(self):
        return convert(self.root/'labels.json', self.root, self.root/'coco.json')[0]

    def test_alias_empty_image_and_nonsquare_coordinates(self):
        data = self.converted()
        self.assertEqual(len(data['images']),2)
        self.assertEqual([a['category_id'] for a in data['annotations']],[1,8])
        ds=DetectionDataset(self.root,self.root/'coco.json')
        img,t=ds[0]
        self.assertEqual(tuple(img.shape),(3,640,640))
        self.assertEqual(t['labels'].tolist(),[0,7])
        for actual,expected in zip(t['boxes'][0].tolist(),[.2025,.205,.205,.21]):
            self.assertAlmostEqual(actual,expected,6)
        self.assertEqual(t['orig_size'].tolist(),[200,100])
        self.assertEqual(tuple(ds[1][1]['boxes'].shape),(0,4))
        flip=DetectionDataset(self.root,self.root/'coco.json',flip=1.)
        self.assertAlmostEqual(float(flip[0][1]['boxes'][0,0]),.7975,6)

    def test_scalabel_single_pixel_box_is_preserved(self):
        self.labels[0]['labels']=[dict(category='person',box2d=dict(x1=20,y1=10,x2=20,y2=10))]
        dump(self.root/'labels.json',self.labels)
        data=self.converted()
        self.assertEqual(data['annotations'][0]['bbox'],[20.,10.,1.,1.])
        self.assertEqual(data['annotations'][0]['area'],1.)

    def test_box_outside_image_is_logged_and_image_is_retained(self):
        self.labels[0]['labels']=[dict(id=4,category='person',box2d=dict(x1=200,y1=10,x2=200,y2=20))]
        dump(self.root/'labels.json',self.labels)
        data=self.converted()
        self.assertEqual(len(data['images']),2)
        self.assertEqual(data['annotations'],[])
        filtered=json.loads((self.root/'coco.json.filtered_boxes.json').read_text())
        self.assertEqual(filtered[0]['image'],'a.jpg')
        self.assertEqual(filtered[0]['label_id'],4)

    def test_unknown_box_category_is_not_silently_dropped(self):
        self.labels[0]['labels'][0]['category']='wrong_dataset'
        dump(self.root/'labels.json',self.labels)
        with self.assertRaises(ValueError):
            self.converted()

    def test_perfect_and_duplicate_predictions(self):
        data=self.converted()
        preds=[dict(image_id=a['image_id'],category_id=a['category_id'],bbox=a['bbox'],score=.9) for a in data['annotations']]
        result=evaluate_predictions(self.root/'coco.json',preds)
        dump(self.root/'metrics.json',result)
        self.assertAlmostEqual(result['mAP50_95'],1.,6)
        self.assertEqual(result['recall'],1.)
        self.assertEqual(result['precision'],1.)
        self.assertIsNone(result['class_wise'][5]['AP'])
        duplicate={**preds[0],'score':.8}
        result=evaluate_predictions(self.root/'coco.json',preds+[duplicate])
        self.assertAlmostEqual(result['precision'],2/3)
        self.assertEqual(result['recall'],1.)

    def test_empty_predictions(self):
        self.converted()
        result=evaluate_predictions(self.root/'coco.json',[])
        self.assertEqual(result['mAP50_95'],0.)
        self.assertEqual(result['recall'],0.)
        self.assertIsNone(result['precision'])

if __name__=='__main__':
    unittest.main()
