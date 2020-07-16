from Decision_Tree.demo1.create_decision_tree import create_tree


def main():
    fr = open('./lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = create_tree(lenses, lenses_labels)
    print(lenses_tree)


if __name__ == "__main__":
    main()
