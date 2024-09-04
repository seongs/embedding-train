import streamlit as st
import os
import json
import pandas as pd

# Set layout to wide mode
st.set_page_config(layout='wide')


def app():
    # 결과를 저장할 데이터프레임을 생성합니다.
    data = {}
    avg_data = {}  # average score를 저장하기 위한 dictionary
    tasks = ['Ko-StrategyQA', 'Markers_bm']
    top_k_types = ['top1', 'top3', 'top5']

    score_types = {
        'top1': ['recall_at_1', 'precision_at_1', 'ndcg_at_1'],
        'top3': ['recall_at_3', 'precision_at_3', 'ndcg_at_3'],
        'top5': ['recall_at_5', 'precision_at_5', 'ndcg_at_5']
    }

    # 각 작업에 대한 데이터를 초기화
    for task in tasks:
        data[task] = {top_k: [] for top_k in top_k_types}

    root_dir = '/data/ONTHEIT/results'

    # 데이터가 저장되어 있는 디렉토리의 모든 하위 폴더를 순회하면서 json 파일을 읽습니다.
    for subdir, dirs, files in os.walk(root_dir):
        if 'data/ONTHEIT' in os.path.relpath(subdir, root_dir):
            continue
        for file in files:
            for task in tasks:
                if file == task + '.json':
                    with open(os.path.join(subdir, file)) as f:
                        d = json.load(f)

                        for top_k in top_k_types:
                            results = {}

                            for score in score_types[top_k]:
                                if 'dev' in d['scores']:
                                    results[score] = d['scores']['dev'][0][score]
                                elif 'test' in d['scores']:
                                    results[score] = d['scores']['test'][0][score]

                            f1_score = 2 * (results[score_types[top_k][1]] * results[score_types[top_k][0]]) / (
                                        results[score_types[top_k][1]] + results[score_types[top_k][0]]) if (results[
                                                                                                                 score_types[
                                                                                                                     top_k][
                                                                                                                     1]] +
                                                                                                             results[
                                                                                                                 score_types[
                                                                                                                     top_k][
                                                                                                                     0]]) > 0 else 0
                            data[task][top_k].append((os.path.relpath(subdir, root_dir), results[score_types[top_k][0]],
                                                      results[score_types[top_k][1]], results[score_types[top_k][2]],
                                                      f1_score))

    # 각 작업에 대해 top1, top3, top5 점수 표시
    for task in tasks:
        st.markdown(f'# {task}')
        for top_k in top_k_types:
            st.markdown(f'## {top_k.capitalize()} Scores')
            df = pd.DataFrame(data[task][top_k],
                              columns=['Subdir', f'Recall_{top_k}', f'Precision_{top_k}', f'NDCG_{top_k}',
                                       f'F1_{top_k}'])
            df = df.sort_values(by=f'NDCG_{top_k}', ascending=False)
            st.dataframe(df, use_container_width=True)

            # 각 모델의 평균 점수를 계산
            for subdir, recall, precision, ndcg, f1 in data[task][top_k]:
                if subdir not in avg_data:
                    avg_data[subdir] = {k: [[], [], [], []] for k in top_k_types}  # 각 top_k에 대해 별도 리스트 생성
                avg_data[subdir][top_k][0].append(recall)
                avg_data[subdir][top_k][1].append(precision)
                avg_data[subdir][top_k][2].append(ndcg)
                avg_data[subdir][top_k][3].append(f1)

    # 각 모델 별로 평균 점수를 계산하고 출력합니다.
    st.markdown('# Average Scores')
    for top_k in top_k_types:
        avg_results = []
        for model in avg_data:
            recall_avg = sum(avg_data[model][top_k][0]) / len(avg_data[model][top_k][0]) if avg_data[model][top_k][
                0] else 0
            precision_avg = sum(avg_data[model][top_k][1]) / len(avg_data[model][top_k][1]) if avg_data[model][top_k][
                1] else 0
            ndcg_avg = sum(avg_data[model][top_k][2]) / len(avg_data[model][top_k][2]) if avg_data[model][top_k][
                2] else 0
            f1_avg = sum(avg_data[model][top_k][3]) / len(avg_data[model][top_k][3]) if avg_data[model][top_k][3] else 0
            avg_results.append([model, recall_avg, precision_avg, ndcg_avg, f1_avg])

        avg_df = pd.DataFrame(avg_results, columns=['Model', f'Average Recall_{top_k}', f'Average Precision_{top_k}',
                                                    f'Average NDCG_{top_k}', f'Average F1_{top_k}'])
        avg_df = avg_df.sort_values(by=f'Average NDCG_{top_k}', ascending=False)
        st.markdown(f'## {top_k.capitalize()} Average Scores')
        st.dataframe(avg_df, use_container_width=True)


if __name__ == "__main__":
    app()

# import streamlit as st
# import os
# import json
# import pandas as pd
#
# # Set layout to wide mode
# st.set_page_config(layout='wide')
#
#
# def app():
#     # baseline 모델들 설정
#     baseline_model_subdirs = [
#         "intfloat/multilingual-e5-large/intfloat__multilingual-e5-large/4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
#         "BAAI/bge-m3/BAAI__bge-m3/5617a9f61b028005a4858fdac845db406aefb181",
#         "Alibaba-NLP/gte-multilingual-base/Alibaba-NLP__gte-multilingual-base/f7d567e1f2493bb0df9413965d144de9f15e7bab"
#     ]
#
#     # 결과를 저장할 데이터프레임을 생성합니다.
#     data = {}
#     avg_data = {}  # average score를 저장하기 위한 dictionary
#     tasks = ['Ko-StrategyQA', 'Markers_bm']
#     top_k_types = ['top1', 'top3', 'top5']
#
#     score_types = {
#         'top1': ['recall_at_1', 'precision_at_1', 'ndcg_at_1'],
#         'top3': ['recall_at_3', 'precision_at_3', 'ndcg_at_3'],
#         'top5': ['recall_at_5', 'precision_at_5', 'ndcg_at_5']
#     }
#
#     # 각 작업에 대한 데이터를 초기화
#     for task in tasks:
#         data[task] = {top_k: [] for top_k in top_k_types}
#
#     root_dir = '/data/ONTHEIT/results'
#
#     # 데이터가 저장되어 있는 디렉토리의 모든 하위 폴더를 순회하면서 json 파일을 읽습니다.
#     for subdir, dirs, files in os.walk(root_dir):
#         if 'data/ONTHEIT' in os.path.relpath(subdir, root_dir):
#             continue
#         for file in files:
#             for task in tasks:
#                 if file == task + '.json':
#                     with open(os.path.join(subdir, file)) as f:
#                         d = json.load(f)
#
#                         for top_k in top_k_types:
#                             results = {}
#
#                             for score in score_types[top_k]:
#                                 if 'dev' in d['scores']:
#                                     results[score] = d['scores']['dev'][0][score]
#                                 elif 'test' in d['scores']:
#                                     results[score] = d['scores']['test'][0][score]
#
#                             f1_score = 2 * (results[score_types[top_k][1]] * results[score_types[top_k][0]]) / (
#                                         results[score_types[top_k][1]] + results[score_types[top_k][0]]) if (results[
#                                                                                                                  score_types[
#                                                                                                                      top_k][
#                                                                                                                      1]] +
#                                                                                                              results[
#                                                                                                                  score_types[
#                                                                                                                      top_k][
#                                                                                                                      0]]) > 0 else 0
#                             data[task][top_k].append((os.path.relpath(subdir, root_dir), results[score_types[top_k][0]],
#                                                       results[score_types[top_k][1]], results[score_types[top_k][2]],
#                                                       f1_score))
#
#     # 각 작업에 대해 top1, top3, top5 점수 표시 및 기준 모델 비교
#     st.markdown('# Models with Higher NDCG than Baseline')
#
#     for top_k in top_k_types:
#         st.markdown(f'## {top_k.capitalize()} - Models with Higher NDCG than Baseline')
#
#         # 기준 모델의 NDCG 점수를 추출
#         baseline_ndcg_scores = {}
#         for task in tasks:
#             baseline_ndcg_scores[task] = []
#             for subdir in baseline_model_subdirs:
#                 for row in data[task][top_k]:
#                     if row[0] == subdir:
#                         baseline_ndcg_scores[task].append(row[3])  # NDCG 스코어 추출
#
#         # 기준 모델의 NDCG 평균 계산
#         baseline_avg_ndcg = {
#             task: sum(baseline_ndcg_scores[task]) / len(baseline_ndcg_scores[task]) if baseline_ndcg_scores[task] else 0
#             for task in tasks}
#
#         # 두 작업에서 모두 baseline보다 높은 모델 필터링
#         higher_ndcg_models = []
#         for subdir, recall, precision, ndcg, f1 in data[tasks[0]][top_k]:
#             ndcg_task1 = ndcg
#             # task2에서 같은 모델의 점수를 찾아 비교
#             for row in data[tasks[1]][top_k]:
#                 if row[0] == subdir:
#                     ndcg_task2 = row[3]
#                     if ndcg_task1 > baseline_avg_ndcg[tasks[0]] and ndcg_task2 > baseline_avg_ndcg[tasks[1]]:
#                         higher_ndcg_models.append([subdir, ndcg_task1, ndcg_task2])
#
#         # 결과를 데이터프레임으로 보여주기
#         if higher_ndcg_models:
#             df = pd.DataFrame(higher_ndcg_models, columns=['Model', f'NDCG_{tasks[0]}', f'NDCG_{tasks[1]}'])
#             st.dataframe(df, use_container_width=True)
#         else:
#             st.markdown('No models found with higher NDCG scores than baseline.')
#
#
# if __name__ == "__main__":
#     app()