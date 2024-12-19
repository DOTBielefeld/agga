import numpy as np

class InstanceSet:
    def __init__(self, instance_set, start_instance_size, set_size=None, instance_increment_size= None, target_reach=None):
        """

        :param instance_set: set of instances available.
        :param start_instance_size: size of the first instances se to be created
        :param set_size: If not set the biggest instance set to be created includes all instances in instance_set. If
        set the biggest instance set will be of the size of the given int.
        """
        self.instance_set = instance_set
        self.start_instance_size = start_instance_size
        self.instance_sets = []
        self.subset_counter = 0

        if set_size and set_size <= len(instance_set):
            self.set_size = set_size
        else:
            self.set_size = len(instance_set)

        if instance_increment_size:
            self.instance_increment_size = instance_increment_size
        else:
            self.instance_increment_size = self.set_size / np.floor(self.set_size/self.start_instance_size)

        if target_reach:
            set_size = self.start_instance_size
            counter = 0
            while set_size <= self.set_size :
                set_size += self.instance_increment_size
                counter += 1
            self.target_increment = np.ceil(target_reach / counter)
        else:
            self.target_increment = 1


    def next_set(self):
        """
        Create a new instance set with instances not included in any created set before. This is not thread safe and
        should only be called from the master node.
        :param id: int. Id of the new instance set. If it is the first set id will be 1.
        """
        # Get instances that were already chosen
        if self.subset_counter == 0:
            seen_instances = []
        else:
            seen_instances = self.instance_sets[self.subset_counter - 1]

        # If we have not chosen any instance before we create a first set
        if not self.instance_sets:
            if self.start_instance_size > len(self.instance_set):
                raise ValueError("The number of instances provided is smaller than the initial instance set size")
            else:
                new_subset = np.random.choice(self.instance_set, self.start_instance_size, replace=False).tolist()
                self.instance_sets.append(new_subset)
        # If we have seen instances before and are still allowed to choose instances we do so
        elif len(seen_instances) <= self.set_size:
            # We either sample as many instances as the slope tells us or the last remainder set to full instance sice
            number_to_sample = int(min(self.instance_increment_size, self.set_size - len(seen_instances)))
            # We can only select instances not chosen before
            possible_instances = [i for i in self.instance_set if i not in seen_instances]
            new_subset = np.random.choice(possible_instances, number_to_sample, replace=False).tolist()
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1] + new_subset)
        # In case we have chosen all instances but still tournaments to run we use the full set of instances for all
        # following tournaments
        else:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])

        self.subset_counter += 1

    def get_subset(self, next_tournament_set_id):
        """
        Create an instance set for the next tournament. The set contains all instances that were included in the
        previous sets as well as a new subset of instances.
        :param next_tournament_set_id: int. Id of the subset to get the next instances for .
        :return: id of the instances set, list containing instances and previous instances of the subset
        """
        assert next_tournament_set_id <= self.subset_counter
        # If we have already created the required subset we return it
        if next_tournament_set_id in range(len(self.instance_sets)):
            next_set = self.instance_sets[next_tournament_set_id]

        elif self.target_increment != 1 and next_tournament_set_id % self.target_increment != 0:
            self.instance_sets.append(self.instance_sets[self.subset_counter - 1])
            next_set = self.instance_sets[next_tournament_set_id - 1]
            self.subset_counter += 1

        # In case we have not, we create the next instance subset
        else:
            self.next_set()
            next_set = self.instance_sets[next_tournament_set_id]
        return next_tournament_set_id, next_set
