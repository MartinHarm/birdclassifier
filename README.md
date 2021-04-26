# BirdClassifier

#Requirements
| Requirement | Version |
| --- | --- |
| `tensorflow` | 2.5.0rc0 |
| `tensorflow_hub` | 0.11.0 |
| `opencv-python` | 4.5.1.48 |
| `numpy` | 1.19.3 |

#Run
| Command | Description |
| --- | --- |
| `py bird_classifier.py` | Start the requests for given images |
| `py bird_classifier_test.py` | Run tests |

#Sample output
```
Run: 1 https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg
1.Match:  Phalacrocorax varius varius with score: 0.8430763483047485
2.Match:  Phalacrocorax varius with score: 0.11654692143201828
3.Match:  Microcarbo melanoleucos with score: 0.024331536144018173
Run: 2 https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg
1.Match:  Galerida cristata with score: 0.8428873419761658
2.Match:  Alauda arvensis with score: 0.08378682285547256
3.Match:  Eremophila alpestris with score: 0.018995527178049088
Run: 3 https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg
1.Match:  Eumomota superciliosa with score: 0.4127245247364044
2.Match:  Momotus coeruliceps with score: 0.05253968387842178
3.Match:  Momotus lessonii with score: 0.048381611704826355
Run: 4 https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg
1.Match:  Aulacorhynchus prasinus with score: 0.8074852824211121
2.Match:  Cyanocorax yncas with score: 0.11162681132555008
3.Match:  Chlorophanes spiza with score: 0.014210059307515621
Run: 5 https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg
1.Match:  Erithacus rubecula with score: 0.8382053971290588
2.Match:  Ixoreus naevius with score: 0.0030795016791671515
3.Match:  Setophaga tigrina with score: 0.002611359115689993
Time spent: 1.9987270832061768

Process finished with exit code 0
```