# embedding-train
**embedding-train** 프로젝트는 **multilingual-e5** 모델을 한국어 데이터로 fine-tuning하여 한국어에 특화된 임베딩 모델을 만드는 것을 목표로 합니다. 이를 통해 한국어 질의와 문서 간의 의미적 유사성을 잘 반영할 수 있는 고성능 임베딩 모델을 구축합니다.

## 프로젝트 버전
- `v1`: InfoNCE loss를 `huggingface Trainer`의 `compute_loss` 함수에 직접 구현하여 학습하는 파이프라인입니다. 이를 통해 기본적인 한국어 임베딩 학습이 가능합니다. [관련 블로그](https://yjoonjang.medium.com/koe5-%EC%B5%9C%EC%B4%88%EC%9D%98-%ED%95%9C%EA%B5%AD%EC%96%B4-%EC%9E%84%EB%B2%A0%EB%94%A9-%EB%AA%A8%EB%8D%B8-multilingual-e5-finetune-22fa7e56d220)
- `v1.1`: InfoNCE loss가 [배치 크기에 큰 영향을 받는다](https://yjoonjang.medium.com/%EB%B0%B0%EC%B9%98-%EC%82%AC%EC%9D%B4%EC%A6%88%EB%A5%BC-%EB%AF%B8%EC%B9%9C%EB%93%AF%EC%9D%B4-%ED%82%A4%EC%9A%B0%EB%8A%94-%EB%B2%95-gradient-cache-60e066907b69)는 사실을 기반으로, **gradient cache**를 적용하여 큰 배치 크기에서도 안정적으로 학습할 수 있도록 개선된 파이프라인입니다.

## 학습

모델을 파인 튜닝하고 학습시키기 위한 단계는 다음과 같습니다:

1. 의존성 패키지를 설치 합니다.
```bash
pip install -r requirements.txt
```
2. `v1.1/scripts/finetune.sh` 스크립트를 수정하여 학습 파라미터를 설정합니다.
3. 설정한 파라미터에 따라 스크립트를 실행하여 모델을 학습시킵니다.
```bash
bash v1.1/scripts/finetune.sh
```

## 데이터 구성
모델 학습에 사용된 데이터는 다음과 같은 구조를 가지고 있습니다:
```json
{
    "query": "사용자가 입력한 질의",
    "document": "질의와 관련된 문서",
    "hard_negative": "질의와 관련성이 적지만, 유사해 보이는 문서"
}
```
데이터는 AIHUB, KorQUAD, KommonGen, Exobrain, KLUE, KoBEST, NIKL로부터 총 10종의 오픈 데이터를 수집하였으며, 총 데이터 통계는 다음과 같습니다.

| **데이터 원천** | **# 샘플**   |
|----------------|-------------|
| AIHUB          | 552,276     |
| KorQuAD        | 28,795      |
| KommonGen      | 14,893      |
| Exobrain       | 78,362      |
| KLUE           | 6,334       |
| KoBEST         | 2,849       |
| NIKL           | 39,919      |
| **Total**      | **723,428** |

## 평가
- `python evaluate.py`명령어로 평가를 수행합니다.
- `streamlit run leaderboard.py`명령어로 평가에 대한 리더보드를 확인합니다.

## 결과
Ko-strategyQA, AutoRAG-embedding-benchmark, PublicHealthQA의 총 3가지 평가 데이터셋으로 평가를 진행했으며, 결과는 다음과 같습니다.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Ko-strategyQA</th>
      <th colspan="4">AutoRAG-benchmark</th>
      <th colspan="4">PublicHealthQA</th>
      <th colspan="4">Avg</th>
    </tr>
    <tr>
      <th>NDCG@1</th><th>F1@1</th><th>NDCG@3</th><th>F1@3</th>
      <th>NDCG@1</th><th>F1@1</th><th>NDCG@3</th><th>F1@3</th>
      <th>NDCG@1</th><th>F1@1</th><th>NDCG@3</th><th>F1@3</th>
      <th>NDCG@1</th><th>F1@1</th><th>NDCG@3</th><th>F1@3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BGE-m3</td>
      <td>75.68</td><td>59.68</td><td>74.79</td><td>55.76</td>
      <td>67.54</td><td>67.54</td><td>47.57</td><td>43.42</td>
      <td>67.53</td><td>67.53</td><td>76.69</td><td>41.56</td>
      <td>70.25</td><td>64.92</td><td>66.35</td><td>46.91</td>
    </tr>
    <tr>
      <td>mGTE</td>
      <td>70.1</td><td>54.91</td><td>70.06</td><td>52.55</td>
      <td>58.77</td><td>58.77</td><td>46.14</td><td>40.35</td>
      <td>58.44</td><td>58.44</td><td>69.06</td><td>38.31</td>
      <td>62.44</td><td>57.37</td><td>61.75</td><td>43.74</td>
    </tr>
    <tr>
      <td>mE5-large</td>
      <td><b>76.86</b></td><td><b>61.07</b></td><td><b>76.48</b></td><td><b>57.05</b></td>
      <td>63.15</td><td>63.15</td><td>44.04</td><td>39.91</td>
      <td>68.83</td><td>68.83</td><td>79.31</td><td><b>42.86</b></td>
      <td>69.61</td><td>64.35</td><td>66.61</td><td>46.61</td>
    </tr>
    <tr>
      <td>kf-deberta</td>
      <td>60.64</td><td>47.19</td><td>61.21</td><td>46.22</td>
      <td>45.61</td><td>45.61</td><td>36.79</td><td>31.58</td>
      <td>54.55</td><td>54.55</td><td>64.69</td><td>35.71</td>
      <td>53.6</td><td>49.12</td><td>54.23</td><td>37.84</td>
    </tr>
    <tr>
      <td><b>KoE5 (Ours)</b></td>
      <td>76.69</td><td>60.70</td><td>75.70</td><td>56.32</td>
      <td><b>70.17</b></td><td><b>70.17</b></td><td><b>48.01</b></td><td><b>44.30</b></td>
      <td><b>71.43</b></td><td><b>71.43</b></td><td><b>80.10</b></td><td><b>42.86</b></td>
      <td><b>72.76</b></td><td><b>67.43</b></td><td><b>67.94</b></td><td><b>47.83</b></td>
    </tr>
  </tbody>
</table>

