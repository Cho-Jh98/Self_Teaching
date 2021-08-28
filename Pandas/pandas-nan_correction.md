1. missing value 가 0 으로 코딩이 되어있는데, 이를 nan 으로 바꾸는 코드를 iterrows 를 이용해서 짜보자. 

```python
def fix_missing(df, col): 
	for i, row in df.iterrows(): 
    val = row[col] 
    if val == 0: 
      df.loc[i, col] = np.nan

출처: https://3months.tistory.com/419 [Deep Play]
```



2. .iloc or .loc 으로 바꾸기

```python
def fix_missing2(df, col):
  for i in df.index:
    val = df.loc[i, col]
    if val == 0:
      df.loc[i, col] = np.nan
```



3. pd.get_value() / pd.set_value()

```python
def fix_missing2(df, col):
  for i in df.index:
    val = df.loc[i, col]
    if val == 0:
      df.set_value(i, col, np.nan)
```



axis = 1 : 가로
axis = 0 : 세로



```python
df.index = ['1', '2', '3', '4' ...]
```

```python
df.set_index(keys, drop = True, append = False, inplace = False, verify_integrity = False)
```

* keys : 인덱스 레이블
* drop : 인덱스로 쓸 열을 데이터 내에서 지울지 여부
* append : 기존에 쓰던 인덱스 삭제 여부
* inplace : 원본 객체를 변경할지 여부
* verify_integrity : 인덱스 중복 여부 체크 -> True : 시간 소요



```python
df.reset_index(level = None, drop = False, inplace = False, col_level = 0, col_fill'')
```

* level = index에서 주어진 단계 제거. default : 모든 단계 제거
* col_level : 멀티 인덱스일 경우 어떤 것으로 삽입할지 설정. default : 0 (첫번째 것)
* col_fill'': 멀티 인덱스의 경우 다른 단계의 이름 설정. default : 0 (이름 없음)