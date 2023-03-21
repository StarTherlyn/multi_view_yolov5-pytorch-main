import os

VOCdevkit_path = 'VOCdevkit'
VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]

for year, image_set in VOCdevkit_sets:
    image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                                 encoding='utf-8').read().strip().split()
    list_file = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s_sv.txt' % (year, image_set)),'w',
                                 encoding='utf-8')

    for image_id in image_ids:
        image_id = image_id[:-1] + '1'
        list_file.write('%s'%(image_id))
        list_file.write('\n')
    list_file.close()