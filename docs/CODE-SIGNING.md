# Code Signing Policy

Free code signing provided by [SignPath.io](https://signpath.io), certificate by [SignPath Foundation](https://signpath.org).

The Windows release (`CosmosCollection.exe`, `CosmosCollection-CLI.exe` and `CosmosCollectionUpdater.exe`) is signed. Windows shows **SignPath Foundation** as the publisher. Only files built from this repository by its [GitHub Actions release workflow](../.github/workflows/build-release.yml) are signed, and every release is approved by hand before it's signed.

## Team Roles

| Role | Members |
|---|---|
| Committers and reviewers | [quake101](https://github.com/quake101) |
| Approvers | [quake101](https://github.com/quake101) |

Committers can change the source code. Reviewers review every change from anyone else, such as pull requests, before it's merged. Approvers approve each signing request.

## Privacy

See the [privacy policy](PRIVACY.md) for what Cosmos Collection sends over the internet, and how to turn it off.
