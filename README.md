# GPUTucker (GPU-based Tucker Decomposition)

For each application, the folders are:
- **include**: All project header files. All third-party header files that do not exist under /usr/local/include are also placed here.
- **lib**: Any libs that get compiled by the project, third-party or any needed in development.
- **source**: The application's source files.

## Usage
GPUTucker requires OpenMP and CUDA libraries.

**Input tensor must follow tab- or space-separated format and base-1 indexing.**

``````
$ ./GPUTucker -i [input_path] -o [order] -r [tucker_rank] -g [num_gpus]
``````
## Program Options
- **-h** or **--help**: Display help menu
- **-o** or **--order**: Input tensor order
- **-r** or **--rank**: Tucker rank (default 10)
- **-g** or **--gpus**: The number of GPUs (default 1)


## Datasets

| Name | Structure | Size | Number of Nonzeros | Download |
| :------------: | :-----------: | :-------------: |:------------:| :------------:|
| NELL-2  | (entity, relation, entity) |  12,092 &times; 9,184 &times; 28,818 | 76,879,419 | [Download](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz) |
| Freebase-music  | (entity, relation, entity) |  23,344,784 &times; 166 &times; 23,344,784  | 99,546,551 | [Download](https://datalab.snu.ac.kr/haten2/freebase_music.tar.gz) |
| DBLP  | (Author, Conference, Year; Count) |   418,236 &times; 3,571 &times; 50  | 1,325,416 | [Download]() |
| Delicious3d  | (user, item, tag) |   532,924 &times; 17,262,471 &times; 2,480,308  | 140,126,181 | [Download](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-3d.tns.gz) |
| Delicious4d  | (user, item, tag, date) |   532,924 &times; 17,262,471 &times; 2,480,308 &times; 1,443 | 140,126,181 | [Download](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-4d.tns.gz) |
| Flickr  | (user, item, tag, date) |   319,686 &times; 28,153,045 &times; 1,607,191 &times; 731 | 112,890,310 | [Download](https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-4d.tns.gz) |