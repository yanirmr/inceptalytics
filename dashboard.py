import streamlit as st
from inceptalytics.analytics import Project
import pandas as pd
import plotly.graph_objects as go


def progress_chart(view, include_empty_files=True, normalize=False):
    counts = view.count(['source_file', 'annotator'], include_empty_files=include_empty_files).unstack()

    if normalize:
        counts = counts.div(counts.max(axis=1), axis=0)  # normalize by files

    return heatmap(counts)


def progress_chart_by_chapter(focus_annos, comments_annos, prototype_annos, include_empty_files=True,
                              normalize=False):
    num_comments_tagged_sentences = dict()
    for label in comments_annos.labels:
        filtered_annos = comments_annos.filter_sentences_by_labels(label)
        num_comments_tagged_sentences[label] = filtered_annos.data_frame.groupby('source_file')['sentence'].nunique()

    sentences_without_focus = num_comments_tagged_sentences["N.F"]
    sentences_with_focus = focus_annos.data_frame.groupby('source_file')['sentence'].nunique()
    short_but_problematic_sentences = num_comments_tagged_sentences["B.C"] + num_comments_tagged_sentences["T.C"] + \
                                      num_comments_tagged_sentences["Overlap"]
    num_prototype_tagged_sentences = dict()
    for label in prototype_annos.labels:
        filtered_annos = prototype_annos.filter_sentences_by_labels(label)
        num_prototype_tagged_sentences[label] = filtered_annos.data_frame.groupby('source_file')['sentence'].nunique()

    body.write(num_prototype_tagged_sentences.keys())
    long_sentences = num_prototype_tagged_sentences["!D"] + num_prototype_tagged_sentences["!C"] + \
                     num_prototype_tagged_sentences["!Q"]
    total_sentences = prototype_annos.data_frame.groupby('source_file')['sentence'].nunique()
    df = pd.concat([sentences_with_focus, sentences_without_focus, short_but_problematic_sentences, long_sentences,
                    total_sentences], axis=1)
    df.columns = ['sentences_with_focus', 'sentences_without_focus', 'short_but_problematic_sentences',
                  'long_sentences', "total_sentences"]
    df = df.fillna(0)
    df = df.astype(int)
    df["sentences_to_be_tagged"] = df.total_sentences - df.sentences_without_focus - df.sentences_with_focus - \
                                   df.short_but_problematic_sentences - df.long_sentences

    body.write(df)


def heatmap(table: pd.DataFrame):
    fig = go.Figure(data=go.Heatmap(
        z=table.values,
        x=table.columns,
        y=table.index,
        colorscale='Blues'))

    fig.update_layout(dict(
        xaxis_title=table.columns.name,
        yaxis_title=table.index.name
    ))

    return fig


def st_grid(rows: int, cols: int = None):
    """
    Creates a rows x cols grid of containers using st.columns and returns it as a nested list of grid cells.

    Args:
        rows: number of rows in the grid.
        cols: number of columns in the grid. If not provided, a square grid is created with cols = rows.
    """
    if cols is None:
        cols = rows

    return [st.columns(cols) for _ in range(rows)]


def st_indexed_grid(row_idxs: list, col_idxs: list = None):
    """
    Creates a grid of containers using st.columns, returning it as a nested dictionary of grid cells.
    Grid cells can be accessed using the provided indices.The first dimension is reserved for rows, the second for
    columns.
    Grid dimensions are implicitly defined by the number of provided indices.

        Args:
            row_idxs: Indices for rows of the grid (1st axis).
            col_idxs: Indices for columns of the grid (2nd axis). If not provided, a square grid is created with
                col_idxs = row_idxs.
    """
    if col_idxs is None:
        col_idxs = row_idxs

    n_rows = len(row_idxs)
    n_cols = len(col_idxs)
    grid = st_grid(n_rows, n_cols)

    indexed_grid = {}
    for i, row_idx in enumerate(row_idxs):
        row = grid[i]
        indexed_row = {col_idx: cell for col_idx, cell in zip(col_idxs, row)}
        indexed_grid[row_idx] = indexed_row

    return indexed_grid


def st_indexed_triangle(row_idxs: list, col_idxs: list = None, offset: int = 0, top=True):
    """
    Creates a in indexed grid from the given indices in which only a triangle is accessible.
    Grid dimensions are implicitly defined by the number of provided indices.

    Args:
        row_idxs: Indices for rows of the grid (1st axis).
        col_idxs: Indices for columns of the grid (2nd axis). If not provided, a square grid is created with
            col_idxs = row_idxs.
        offset: Offset from the main diagonal to use for the triangle. If an
        top: If set to true, the top triangle is used, if set to false, the bottom triangle is used.
    """
    if offset < 0:
        raise ValueError('Offset must be a positive integer.')

    if col_idxs is None:
        col_idxs = row_idxs

    n_rows = len(row_idxs) - offset
    n_cols = len(col_idxs) - offset
    grid = st_grid(n_rows, n_cols)

    if top:
        row_slicer = slice(0, n_rows)
        col_slicer = slice(offset, None)
    else:
        row_slicer = slice(offset, None, 1)
        col_slicer = slice(n_cols, None, -1)

    indexed_grid = {}
    for i, row_idx in enumerate(row_idxs[row_slicer]):
        row = grid[i]
        indexed_row = {col_idx: cell for col_idx, cell in zip(col_idxs[col_slicer], row)}
        indexed_grid[row_idx] = indexed_row

    # TODO: Map bottom half of triangle to top half (if top), vice versa (if bottom).

    return indexed_grid


def get_focus_dist_per_sentence(layer_annos):
    num_tagged_sentences = dict()
    for label in layer_annos.labels:
        filtered_annos = layer_annos.filter_sentences_by_labels(label)
        num_tagged_sentences[label] = len(filtered_annos.data_frame['sentence'].unique())
    return num_tagged_sentences


@st.cache
def load_project():
    return Project.from_remote(project='this-american-life',
                               remote_url='http://harp.wisdom.weizmann.ac.il:8080/',
                               auth=('yanir', 'yanir'))


st.set_page_config("Inception Analytics", None, "wide", "auto")

body, stats = st.columns([4, 1])

body.write(
    """
    # Inception Analytics
    
    More than you ever wanted to know about your annotation project
    """
)

project = Project.from_remote(project='this-american-life',
                              remote_url='http://harp.wisdom.weizmann.ac.il:8080/',
                              auth=('yanir', 'yanir'))

if project:
    ## if you already know which layers you are interested in, you may want to hard-code them here.
    ## especially if you are using just some elements of the build-in layers 
    layers = []
    if project.custom_layers:
        layers = sorted(project.custom_layers)
    else:
        layers = sorted(project.layers)

    layer = st.sidebar.selectbox(
        'Select Layer',
        layers,
        format_func=lambda x: x.split('.')[-1]
    )

    feature = st.sidebar.selectbox(
        'Select Feature',
        sorted(project.features(layer))
    )

    annotators = sorted(project.annotators)
    if len(annotators) > 0:
        selected_annotators = st.sidebar.expander('Select Annotators').multiselect(
            "Annotators",
            options=annotators,
            default=["alona", "smadar", "tirza"],
            key="annotator_select",
        )
    else:
        st.warning('No annotators found in the project.')
        st.stop()

    files = sorted(project.source_file_names)
    if len(files) > 0:
        selected_files = st.sidebar.expander('Select Source Files').multiselect(
            "Files",
            options=files,
            default=list(files),
            key="files_select",
        )
    else:
        st.warning('No source files found in the project.')
        st.stop()

    view = project.select(
        annotation=project.feature_path(layer, feature),
        annotators=selected_annotators,
        source_files=selected_files
    )

    nr_of_annotated_files = len(project.source_file_names)



    body.write('## Focus Annotation Overview')
    body.write('### How many sentences have been tagged for focus?')
    focus_layer = 'webanno.custom.Focus1'
    feature = 'Focus'
    feature_path = f'{focus_layer}>{feature}'
    focus_annos = project.select(annotation=feature_path)
    num_focus_tagged_words = len(focus_annos.data_frame['sentence'])
    body.write(f'The number of focused words:  {num_focus_tagged_words}')
    num_focus_tagged_sentences = len(focus_annos.data_frame['sentence'].unique())
    body.write(f'The number of sentences with one or more focused words:  {num_focus_tagged_sentences}')

    comments_layer = 'webanno.custom.Comments'
    feature = 'Comments'
    feature_path = f'{comments_layer}>{feature}'
    comments_annos = project.select(annotation=feature_path)
    num_comments_tagged_sentences = dict()
    for label in comments_annos.labels:
        filtered_annos = comments_annos.filter_sentences_by_labels(label)
        num_comments_tagged_sentences[label] = len(filtered_annos.data_frame['sentence'].unique())
    body.write(f'The number of sentences without focused words: {num_comments_tagged_sentences["N.F"]}')

    body.write(focus_annos.data_frame.groupby('source_file')['sentence'].nunique())
    prototype_layer = 'webanno.custom.Prototype'
    feature = 'Prototype'
    prototype_layer_feature_path = f'{prototype_layer}>{feature}'
    prototype_annos = project.select(annotation=prototype_layer_feature_path)
    progress_chart_by_chapter(focus_annos, comments_annos, prototype_annos)

    body.write('Select details to include in the overview')

    # split_by_files = body.checkbox('Files')
    # split_by_labels = body.checkbox('Labels')

    count_fn = view.value_counts  # if split_by_labels else view.count
    levels = ["source_file"]
    count_overview = count_fn(grouped_by=levels)

    if count_overview.index.nlevels > 1:
        count_overview = count_overview.unstack(level=0)

    count_overview = count_overview.fillna(0).astype(int)
    body.write(count_overview.T)

    body.write('## Focus Progress')

    focus_layer = 'webanno.custom.Focus1'
    feature = 'Focus'
    focus_layer_feature_path = f'{focus_layer}>{feature}'
    focus_annos = project.select(annotation=focus_layer_feature_path)

    show_percentages_per_file = body.checkbox('Show completion status of files.')
    body.write(progress_chart(view, normalize=show_percentages_per_file))

    body.write(f"## Current View:  **{feature}**")
    # body.write(f'Selected Annotation: **{feature}**')
    body.write('Selected annotators: ' + str(len(selected_annotators)))
    body.write('Selected files: ' + str(len(selected_files)))
    counts = view.count(['annotator', 'source_file'])
    body.write(f'Annotations in Selection: {sum(counts)}')

    body.write('### Breakdown by Annotator')
    body.write(view.count('annotator'))
    body.write('### Breakdown by Labels')
    body.write(view.value_counts())

    # if len(view.annotators) == 1:
    #     st.info('Only one annotator provided annotations for this label. '
    #             'Agreement statistics and confusion matrices are therefore omitted.')
    #     st.stop()

    # body.write('## Agreement Statistics')
    # iaa = view.iaa(measure=iaa_type)
    # body.metric(label=iaa_type, value=str(np.round(iaa, 4)))
    # body.write(view.pairwise_kappa())
    #
    # body.write('### IAA by Label')
    #
    # individual_iaa_labels = body.multiselect(
    #     'Labels to calculate IAA',
    #     options=view.labels,
    #     default=view.labels,
    #     key='individual_iaa_labels'
    # )
    #
    # if individual_iaa_labels:
    #     if iaa_type == 'gamma':
    #         iaa_label_scores = [view.filter_labels(label).iaa(measure=iaa_type) for label in individual_iaa_labels]
    #         body.write(pd.Series(iaa_label_scores, index=individual_iaa_labels, name=iaa_type))
    #     else:
    #         body.warning('Unitising IAA per Label is currently only supported for the _gamma_ IAA measure. '
    #                      'Please select the appropriate IAA statistic on the left. '
    #                      'Keep in mind that calculating the _gamma_ measure per label may take a while.')
    #
    # body.write('## Confusion Matrices')
    # only_differences = body.checkbox('Display only differences', False)
    #
    # if len(annotators) > 2:
    #     body.write('### Total Confusion Matrix')
    #     cms_total = view.confusion_matrices(only_differences, aggregate='total')
    #     body.write(heatmap(cms_total))
    #
    # body.write('### Individual Confusion Matrices')
    # by_annotator = body.checkbox('Aggregate by annotators', False)
    #
    # cms = view.confusion_matrices(only_differences, aggregate='by_annotator' if by_annotator else None)
    # conf_plots = cms.apply(heatmap)

    # # TODO: cleanup
    # # display a list of matrices
    # if by_annotator:
    #     for plot in conf_plots:
    #         body.write(plot)
    # # find good layout to present pairwise matrices
    # else:
    #     max_cols = body.number_input('Maximum Number of Columns', min_value=1, value=4)
    #     if len(project.annotators) <= max_cols:  # organise matrices in triangle
    #         grid = st_indexed_triangle(project.annotators, offset=1)
    #         for idx, plot in conf_plots.iteritems():
    #             a, b = idx
    #             grid[a][b].write(plot)
    #     else:  # organise plots in grid, square if possible
    #         n_annotators = len(annotators)
    #         n_cols = min(ceil(n_annotators ** 0.5), max_cols)
    #         n_rows = ceil(n_annotators / n_cols)
    #         grid = st_grid(n_rows, n_cols)
    #
    #         i = j = 0
    #         for plot in conf_plots:
    #             grid[i][j].write(plot)
    #             j = (j + 1) % n_cols
    #             if j == 0:
    #                 i = (i + 1) % n_rows
