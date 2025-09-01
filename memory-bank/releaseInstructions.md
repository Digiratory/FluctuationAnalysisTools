# StatTools Release Instructions

## Overview
StatTools uses automated GitHub Actions for releases. Version management is handled by Git tags with setuptools-scm. The release process is triggered automatically when a release tag is pushed.

## Version Management
- **Source**: Version determined by Git tags in format `release/X.Y.Z`
- **Automation**: setuptools-scm extracts version from Git history
- **Pattern**: `release/1.9.0`, `release/1.8.0`, etc.

## Automated Release Process

### 1. Pre-Release Checklist
- [ ] All tests pass locally: `pytest`
- [ ] Code quality checks pass: `pre-commit run --all-files`
- [ ] Documentation is updated
- [ ] Changelog is updated in `CHANGELOG.md`
- [ ] No breaking changes without proper deprecation warnings
- [ ] Dependencies are up to date and tested

### 2. Prepare Release Branch
```bash
# Create release branch from main
git checkout main
git pull origin main
git checkout -b release/v1.9.1
```

### 3. Update Documentation
- Update `CHANGELOG.md` with release notes
- Ensure `README.md` reflects current features
- Update any version-specific documentation
- Review and update memory bank if needed

### 4. Run Local Tests
```bash
# Run complete test suite
pytest tests/ -v

# Check code quality
pre-commit run --all-files

# Test installation
pip install -e .
```

### 5. Create and Push Release Tag
```bash
# Create annotated tag (this triggers automated release)
git tag -a release/1.9.1 -m "Release version 1.9.1

New features:
- [List major new features]

Bug fixes:
- [List important bug fixes]

Performance improvements:
- [List optimizations]

Breaking changes:
- [List if any, with migration notes]"

# Push tag to trigger automated release
git push origin release/1.9.1
```

### 6. Monitor Automated Release
The `test-and-release.yml` workflow will automatically:
- Run tests on Ubuntu, Windows, and macOS
- Build wheels for all platforms
- Create GitHub release with artifacts
- Publish to PyPI

### 7. Post-Release Verification
- [ ] Verify GitHub release was created
- [ ] Check PyPI for new version
- [ ] Test installation: `pip install StatTools==1.9.1`
- [ ] Monitor for any immediate issues
- [ ] Update memory bank with new version information

## Version Numbering Guidelines

### Major Version (X.0.0)
- Breaking API changes
- Major architectural changes
- Significant scope changes

### Minor Version (1.X.0)
- New backward-compatible features
- Significant improvements
- New analysis methods or generators

### Patch Version (1.9.X)
- Bug fixes
- Small improvements
- Documentation updates
- Security fixes

## Branching Strategy
- `main`: Primary development branch
- `release/vX.Y.Z`: Release preparation (optional, can tag directly from main)
- Feature branches: `feature/description`
- Bug fix branches: `fix/issue-description`

## Troubleshooting

### Release Fails
1. **Tests fail**: Fix failing tests before retrying
2. **Build fails**: Check C++ compilation and dependencies
3. **PyPI upload fails**: Verify PyPI token in repository secrets
4. **Tag issues**: Ensure tag format matches `release/X.Y.Z`

### Rollback Procedure
```bash
# Delete tag locally and remotely
git tag -d release/1.9.1
git push origin :refs/tags/release/1.9.1

# If already on PyPI, yank the release
# Contact PyPI admins for removal if necessary
```

## Quality Gates
- **Automated Tests**: Must pass on all platforms (Ubuntu, Windows, macOS)
- **Code Quality**: pre-commit checks must pass
- **Documentation**: Changelog and release notes must be complete
- **Dependencies**: All dependencies must be available and compatible

## Release Frequency
- **Regular releases**: Every 1-2 months for minor versions
- **Patch releases**: As needed for critical fixes
- **Major releases**: When significant changes warrant it

## Communication
- **Automated**: GitHub release notes are created automatically
- **Manual**: Update relevant communities and users for major releases
- **Issues**: Monitor GitHub issues post-release

## Maintenance Notes
- Keep PyPI token secure in repository secrets
- Regularly update GitHub Actions versions
- Monitor workflow success rates
- Update dependencies periodically
