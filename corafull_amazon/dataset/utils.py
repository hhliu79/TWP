import collections

class semi_task_manager:
    def __init__(self, dataset_info):
        self.task_info = collections.OrderedDict()
        self.n_labels_per_task = dataset_info['n_labels_per_task']
        self.n_tasks = dataset_info['n_tasks']
        self.labels_of_tasks = dataset_info['labels_of_tasks']
        self.g = None
        self.newg = []
        self.degree = None

    def add_task(self, task_i, masks):
        self.task_info[task_i] = masks

    def retrieve_task(self, task_i):
        """Retrieve information for task_i
        """
        return self.task_info[task_i]

    def old_tasks(self):
        return self.task_info.keys()

    def get_label_offset(self, task_i):
        return task_i * self.n_labels_per_task, (task_i + 1) * self.n_labels_per_task 

    def add_g(self, g):
        self.g  = g
    
    def add_newg(self, newg):
        self.newg.append(newg)
        
    def add_degree(self, degree):
        self.degree = degree


class task_manager:
    def __init__(self):
        pass