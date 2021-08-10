from detectron2.distiller_models import model_dict

class Distill():
    def get_teacher_name(self, model_path):
        """parse teacher name"""
        segments = model_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]

    def load_teacher(self, model_path, n_cls):
        print('==> loading teacher model')
        model_t = get_teacher_name(model_path)
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
        return model