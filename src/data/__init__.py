from importlib import import_module
# from dataloader import MSDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train
    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)
class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                m = import_module('data.' + args.data_train[0].lower())
                # datasets.append(getattr(m, module_name)(args, name=d))
                datasets.append(getattr(m, args.data_train[0])(args, name=d))
            # print("module_name.lower():",module_name.lower())
            # self.loader_train = MSDataLoader(
            #     args,
            self.loader_train = DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )
        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100','Manga109']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)
            # print("test: module_name.lower():", module_name.lower())
            # self.loader_test.append(MSDataLoader(
            #     args,
            self.loader_test.append(DataLoader(
                testset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=not args.cpu
            ))
        self.loader_controller_train = None
        testset=[]
        for d in ['DIV2K']:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset.append(getattr(m, module_name)(args,train=False, name=d))
                # testset = getattr(m, module_name)(args, train=False, name=d)
            # print("test: module_name.lower():", module_name.lower())
            # self.loader_controller_train = MSDataLoader(
            #     args,
            self.loader_controller_train = DataLoader(
                MyConcatDataset(testset),
                # testset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

