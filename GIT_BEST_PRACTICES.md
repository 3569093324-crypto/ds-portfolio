# Git 深度使用指南

> Day 53 | Phase 5: 工程化与作品集打磨

---

## 1. Conventional Commits 规范

```
<type>: <short description>

[optional body]

[optional footer]
```

### Types:

| Type | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: add SHAP waterfall plot` |
| `fix:` | Bug 修复 | `fix: handle NaN in missing indicator` |
| `docs:` | 文档更新 | `docs: add model comparison section` |
| `refactor:` | 代码重构 | `refactor: split cleaning into SRP functions` |
| `test:` | 测试 | `test: add edge case tests for clean_data` |
| `style:` | 代码风格 | `style: apply black formatter` |
| `perf:` | 性能优化 | `perf: vectorize distance matrix calculation` |
| `chore:` | 杂项 | `chore: update .gitignore` |

### 好的 vs 不好的 Commit Message

```
❌ "fix"
❌ "update code"
❌ "asdf"
❌ "wip"

✅ "feat: implement XGBoost with early stopping (AUC +0.03)"
✅ "fix: prevent data leakage in StandardScaler pipeline"
✅ "docs: add business value analysis with ROI calculations"
✅ "refactor: extract _fill_missing_numeric into SRP function"
```

---

## 2. 常用 Git 命令

### 日常工作流
```bash
git status                  # 查看当前状态
git diff                    # 查看未暂存的更改
git add <file>              # 暂存文件
git commit -m "feat: ..."   # 提交
git push origin main        # 推送
git pull origin main        # 拉取
```

### 分支管理
```bash
git branch feature-xyz      # 创建分支
git checkout feature-xyz    # 切换分支
git checkout -b feature-xyz # 创建并切换
git merge feature-xyz       # 合并分支
```

### 历史整理
```bash
git log --graph --oneline --all  # 查看所有分支历史图
git rebase -i HEAD~5             # 交互式rebase最近5个commit
git stash                        # 暂存当前工作
git stash pop                    # 恢复暂存
git cherry-pick <commit-hash>    # 将特定commit应用到当前分支
```

### 回退
```bash
git reset --soft HEAD~1    # 撤销commit, 保留更改
git reset --hard HEAD~1    # 完全回退到上一个commit
git revert <commit-hash>   # 创建一个新commit来撤销
```

---

## 3. .gitignore 最佳实践

已在仓库根目录创建 `.gitignore`，涵盖：
- Python 缓存文件 (`__pycache__/`, `*.pyc`)
- 虚拟环境 (`venv/`, `.venv/`)
- IDE 配置 (`.vscode/`, `.idea/`)
- 数据库文件 (`*.db`, `*.sqlite`)
- 数据文件 (`*.csv`, `*.json`, `*.parquet`)
- OS 文件 (`.DS_Store`, `Thumbs.db`)

---

## 4. GitHub Profile 优化

### Pin 项目
在 GitHub Profile 页面 Pin 4 个项目：
1. `project_1` — 商业数据分析报告
2. `project_2` — Streamlit 交互式看板
3. `project_3` — A/B 测试实验设计
4. `project_4` — 用户复购预测 ML 项目

### README Badges
每个项目 README 都已添加 Badges：
```markdown
![Phase](https://img.shields.io/badge/Phase-4%20ML-blue)
![Status](https://img.shields.io/badge/Status-Complete-green)
```

---

## 5. 面试中的 Git 问题

**Q: "你的 Git 工作流是什么？"**

> A: "我遵循 Conventional Commits 规范（feat/fix/docs等）。
> 每个功能在独立分支开发，通过 PR 合并到 main。
> Commit 粒度控制在每个逻辑步骤一个 commit。
> 使用 rebase 保持提交历史整洁。"

**Q: "遇到 merge conflict 怎么解决？"**

> A: "首先 git status 查看冲突文件，编辑文件解决冲突标记（<<<<<<<），
> 然后 git add 标记为已解决，最后 git commit 完成合并。
> 如果是复杂冲突，用 git mergetool 或 IDE 的三路合并视图辅助。"
