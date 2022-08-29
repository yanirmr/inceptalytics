"""
# INCEpTALYTICS Quick Start Guide
The _INCEpTALYTICS_ package allows you to export and analyse annotation projects using the [_INCEpTION_](https://inception-project.github.io/) annotation tool. 
This guide gives an overview over its functionalities. It assumes that you are familiar with the INCEpTION annotation tool.

## Loading a Project
Projects can be loaded in two ways: 

* Importing a [zipped XMI](https://inception-project.github.io/releases/22.4/docs/user-guide.html#sect_formats_uimaxmi) export
* INCEpTION's [remote API](https://inception-project.github.io/releases/22.5/docs/admin-guide.html#sect_remote_api).

***NOTE***: XMI exports must be in XMI v1.0 format, as INCEpTALYTICS is built on top of [_dkpro cassis_](https://github.com/dkpro/dkpro-cassis).
"""

from inceptalytics import Project

# project = Project.from_zipped_xmi('C:\\Users\\User\\Dropbox\\Projects\\segmenter\\data\\tagged_tal.zip')

project = Project.from_remote(project='this-american-life',
                              remote_url='http://harp.wisdom.weizmann.ac.il:8080/',
                              auth=('yanir', 'yanir'))  # TODO: replace with "remote" credentials

"""Once a project is loaded, you can access different properties such as annotators and annotated files. 
Per default, those include only annotators who annotated at least a single document and documents that contain at least a single annotation.
"""

print('Annotators:', project.annotators)
print('Files:', project.source_file_names)
print('Layers:', project.layers)

"""You can also access the typesystem and CAS objects directly. There is a single CAS object per source file.

See the [dkpro cassis documentation](https://cassis.readthedocs.io/en/latest/) for more details on their usage.
"""

typesystem = project.typesystem
cas_objects = project.cas_objects

# do something useful with those here

"""## Analysing a specific annotation

Annotations are organised in _layers_ and _features_. To analyse a specific annotation, you need to select a feature-layer combination. The returned _View_ offers the main analysis functionalities of INCEpTALYTICS.
"""

# count focus annotations and sentences with focus annotations

focus_layer = 'webanno.custom.Focus1'
print(f'Features: {project.features(focus_layer)}')
feature = 'Focus'
focus_layer_feature_path = f'{focus_layer}>{feature}'
focus_annos = project.select(annotation=focus_layer_feature_path)

sentence_layer = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
feature = 'id'
sentence_feature_path = f'{sentence_layer}>{feature}'
sentence_annos = project.select(annotation=sentence_feature_path)

num_total_sentences = len(sentence_annos.data_frame['sentence'].unique())
num_focus_tagged_sentences = len(focus_annos.data_frame['sentence'].unique())

print(f'Total sentences: {num_total_sentences}')
print(f'Focus tagged sentences: {num_focus_tagged_sentences}')
print(f'Focus tagged words: {focus_annos.count()}')

comments_layer = 'webanno.custom.Comments'
print(f'Features: {project.features(comments_layer)}')
feature = 'Comments'
comments_layer_feature_path = f'{comments_layer}>{feature}'
comments_annos = project.select(annotation=comments_layer_feature_path)

num_tagged_sentences = dict()
for label in focus_annos.labels:
    filtered_annos = focus_annos.filter_sentences_by_labels(label)
    num_tagged_sentences[label] = len(filtered_annos.data_frame['sentence'].unique())

print(num_tagged_sentences)

"""You can also create a view that contains a specific subset of files and annotators."""

# reduced_pos_annos = project.select(annotation=feature_path,
#                                    annotators=['ia-test1', 'ia-test2'],
#                                    source_files=['test1.txt', 'test2.txt'])

"""Once we have selected a specific annotation, we can look at some numbers, e.g. the total number of annotations."""

print('# pos annotations in view:', comments_annos.count())
print('# focus annotations in view:', focus_annos.count())

"""Many methods of the View API accept the `grouped_by` parameter. We can use it to refine a query and organise returned values."""

print('# annotations per file per annotator', comments_annos.count(grouped_by=['source_file', 'annotator']))
print('label distribution', comments_annos.count(grouped_by='annotation'))

print('# annotations per file per annotator', focus_annos.count(grouped_by=['source_file', 'annotator']))
print('label distribution', focus_annos.count(grouped_by='annotation'))

"""Most methods of the View API return [pandas](https://pandas.pydata.org/) objects. In case you want to implement an analysis not covered by the API, you can directly work with a `DataFrame` as well."""

df_pos = comments_annos.data_frame
df_pos.head()

df_focus = focus_annos.data_frame
df_focus.head()
"""If you want to use your annotated data for ML training, INCEpTALYTICS can do a simple majority vote.
The `levels` parameter controls across which unit levels annotations are aggregated. `['sentence', 'begin', 'end']` aggregates over individual spans contained in a sentence.
"""

dataset_pos = comments_annos.consolidated_annotations(levels=['sentence', 'begin', 'end'])
dataset_pos.head()

dataset_focus = focus_annos.consolidated_annotations(levels=['sentence', 'begin', 'end'])
dataset_focus.head()

"""The resulting DataFrame can be stored in many common formats."""

# dataset.to_csv('../data/focus_annos.csv')

"""### Inspecting Data in Detail

To get an overview over the annotation, you can look at the document-annotator matrix.
"""
comments_annos.document_annotator_matrix

# focus_annos.document_annotator_matrix

"""It may be useful to have a look at the text that was annotated to adjudicate disagreements."""
#
# document_annotator_matrix = pos_annos.document_annotator_matrix
# covered_texts = pos_annos.texts
# document_annotator_matrix.join(covered_texts).head()
#
# """If you are looking for annotation quality, we also provide confusion matrices and agreement measures. Confusion matrices produced pairwise and are indexed by annotators."""
#
# cms = pos_annos.confusion_matrices()
# cms[('ia-test3', 'ia-test2')]
#
# """If you are not interested in individual annotators, but overall disagreements over certain classes, you can aggregate the pairwise matrices into a single matrix."""
#
# # sum over all pairwise matrices
# print(pos_annos.confusion_matrices(aggregate='total'))
#
# """If you want to quantify disagreements, INCEpTALYTICS offers different agreement measures which can be calculated pairwise or in aggregate."""
#
# print('Krippendorff\'s alpha: ', pos_annos.iaa())
# print('Pairwise Cohen\'s Kappa: ', pos_annos.iaa_pairwise(measure='kappa'))
