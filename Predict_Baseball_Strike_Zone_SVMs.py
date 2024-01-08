import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

def strike_zone(df):

  df['type'] = df['type'].map({'S':1,'B':2})
  # print(aj.type)

  df = df.dropna(subset =['plate_x','plate_z','strikes','type'])

  plt.scatter(x = df.plate_x, y = df['plate_z'], c = df['type'],cmap=plt.cm.coolwarm, alpha = 0.25)

  training_set,validation_set = train_test_split(df, random_state=1)
  classifier = SVC(gamma=.8,C=.8)
  svc_fit = classifier.fit(training_set[['plate_x','plate_z','strikes']],training_set.type)

  # draw_boundary(ax, svc_fit) ##only works with two features
  # ax.set_ylim(-2,6)
  # ax.set_xlim(-3,3)
  # plt.show()

  print(classifier.score(validation_set[['plate_x','plate_z','strikes']],validation_set.type))

strike_zone(aaron_judge)
strike_zone(jose_altuve)
strike_zone(david_ortiz)