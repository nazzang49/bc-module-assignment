# (Boost Camp AI Tech U-Stage) Practice Modulization
## Branch
- `main` : 최상위 디폴트 브랜치
- `develop` : 각 `feature` 브랜치 병합을 위한 베이스 브랜치
- `feature` : 개인 작업 브랜치 (`feature/T2096`)

## Initialization
1. Git Bash 열기
2. Clone 원하는 경로로 이동
```
git clone https://github.com/nazzang49/bc-module-assignment.git
cd bc-module-assignment
git branch (브랜치 확인)
git flow init (선택사항 => 명령어 입력 후 Enter 쭉-)
git checkout develop
git branch feature/T2096 (본인 캠퍼 아이디)
git branch (작업 브랜치 생성 확인)
git checkout feature/T2096
git push --set-upstream origin feature/T2222 (remote 최초 업로드)
```
3. 위 단계까지 마무리 하면 `main` `develop` `feature/T2096` 모두 `Local = Remote` 형상 싱크 완료
4. Github 에서 작업 브랜치 생성되었는지 확인

## Commit and Push
1. 본인 `feature/T2096` 작업 브랜치에서 단위 작업 완료 시 Commit and Push
2. `develop` 병합 => 의논 후 진행
```
git add .
git commit -m 'init'
git push origin
```

## (Optional) GUI
1. SourceTree
2. GitKraken
3. GitDesktop

## (Optional) Source Diff
1. Github
2. [ArcroDiff](http://www.acrosoft.pe.kr/acroedit/)

## Git Docs
https://git-scm.com/book/ko/v2
