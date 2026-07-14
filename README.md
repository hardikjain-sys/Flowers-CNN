# Flowers-CNN
Written in PyTorch<br>
To run the code properly the structure should be:
 
FlowerCNN/<br>
├── flower_photos/      
├── testPhotos/     
├── main.py


link for flower_photos - [http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

link for testPhotos - [https://drive.google.com/file/d/1YdwfMu1XfISnUhVK-qvXzuHuRby0CNsL/view?usp=sharing](https://drive.google.com/file/d/1YdwfMu1XfISnUhVK-qvXzuHuRby0CNsL/view?usp=sharing)

Output - <br>
```bash
Epoch 1/25, Train Loss 1.4305048338744952, Train Acc 39.8841961852861%,Val Loss 1.1336410278859346, Val Acc 53.67847411444142%
Epoch 2/25, Train Loss 1.2160903908636258, Train Acc 50.0%,Val Loss 0.9901018557341202, Val Acc 58.991825613079016%
Epoch 3/25, Train Loss 1.1257290101569633, Train Acc 54.495912806539515%,Val Loss 0.9724844590477322, Val Acc 61.03542234332425%
Epoch 4/25, Train Loss 1.0644290745258331, Train Acc 57.93596730245232%,Val Loss 0.868467610815297, Val Acc 67.8474114441417%
Epoch 5/25, Train Loss 0.9651639882637106, Train Acc 62.874659400544964%,Val Loss 0.7977873980998993, Val Acc 68.66485013623979%
Epoch 6/25, Train Loss 0.9275493349718011, Train Acc 64.57765667574932%,Val Loss 0.8064317210860874, Val Acc 69.75476839237056%
Epoch 7/25, Train Loss 0.8803199866543645, Train Acc 65.56539509536785%,Val Loss 0.7744154139705326, Val Acc 70.57220708446866%
Epoch 8/25, Train Loss 0.8443069360826326, Train Acc 67.64305177111717%,Val Loss 0.7061264178027278, Val Acc 73.70572207084469%
Epoch 9/25, Train Loss 0.8158779941175295, Train Acc 69.07356948228883%,Val Loss 0.7416352422341056, Val Acc 73.84196185286103%
Epoch 10/25, Train Loss 0.7879814476422642, Train Acc 69.82288828337875%,Val Loss 0.6629746037980785, Val Acc 74.65940054495913%
Epoch 11/25, Train Loss 0.7458571030393891, Train Acc 71.04904632152589%,Val Loss 0.6533829012642736, Val Acc 74.52316076294278%
Epoch 12/25, Train Loss 0.7207626966030701, Train Acc 71.93460490463215%,Val Loss 0.6235149392615194, Val Acc 76.97547683923706%
Epoch 13/25, Train Loss 0.6887363292600798, Train Acc 73.9100817438692%,Val Loss 0.6921876953995746, Val Acc 72.75204359673025%
Epoch 14/25, Train Loss 0.6828981165004813, Train Acc 74.08038147138964%,Val Loss 0.6164561561916185, Val Acc 75.47683923705722%
Epoch 15/25, Train Loss 0.6257604541985885, Train Acc 76.3283378746594%,Val Loss 0.6201034661220468, Val Acc 77.9291553133515%
Epoch 16/25, Train Loss 0.6294739754951518, Train Acc 77.04359673024524%,Val Loss 0.5817371360633684, Val Acc 76.02179836512262%
Epoch 17/25, Train Loss 0.5812918443394743, Train Acc 78.26975476839237%,Val Loss 0.587663163309512, Val Acc 76.97547683923706%
Epoch 18/25, Train Loss 0.5648281635797542, Train Acc 79.59809264305177%,Val Loss 0.5548729728097501, Val Acc 79.56403269754769%
Epoch 19/25, Train Loss 0.549858081114033, Train Acc 79.70027247956402%,Val Loss 0.6327810650286467, Val Acc 78.88283378746594%
Epoch 20/25, Train Loss 0.5366925189028615, Train Acc 80.41553133514986%,Val Loss 0.6307976356019145, Val Acc 77.11171662125341%
Epoch 21/25, Train Loss 0.5343243625500927, Train Acc 80.44959128065395%,Val Loss 0.5850196085546328, Val Acc 78.33787465940054%
Epoch 22/25, Train Loss 0.5000287023899348, Train Acc 80.79019073569482%,Val Loss 0.5717470943927765, Val Acc 78.7465940054496%
Epoch 23/25, Train Loss 0.47546554437798, Train Acc 82.83378746594006%,Val Loss 0.6158358804557634, Val Acc 79.01907356948229%
Epoch 24/25, Train Loss 0.4643282310470291, Train Acc 83.48092643051771%,Val Loss 0.6788008906271147, Val Acc 77.9291553133515%
Epoch 25/25, Train Loss 0.42442890062280325, Train Acc 84.43460490463215%,Val Loss 0.614512054492598, Val Acc 79.291553133515%
Model saved
 True: daisy, Predicted daisy
 True: daisy, Predicted roses
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: daisy, Predicted daisy
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted daisy
 True: dandelion, Predicted sunflowers
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: dandelion, Predicted dandelion
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted tulips
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: roses, Predicted roses
 True: sunflowers, Predicted dandelion
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted dandelion
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: sunflowers, Predicted sunflowers
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips
 True: tulips, Predicted tulips

 unseen images accuracy: 90.0%
```
