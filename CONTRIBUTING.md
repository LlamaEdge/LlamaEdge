# Contributing guidelines

## How to become a contributor and submit your own code

### Developer Certificate of Origin (DCO)

We'd love to accept your patches! Before we can take them, you are often asked to sign a DCO (Developer Certificate of Origin) to ensure that the project has the proper rights to use your code. [A Complete Guide to DCO for Open Source Developers](https://www.secondstate.io/articles/dco/) tells you how to do it. Please read it carefully before you start your work.

### Git

Please check out a recent version of `dev` branch before starting work, and rebase onto `dev` before creating a pull request.
This helps keep the commit graph clean and easy to follow. In addition, please sign off each of your commits.

### GitHub Issues

If you want to work on a GitHub issue, check to make sure it's not assigned to someone first.
If it's not assigned to anyone, assign yourself once you start writing code.
(Please don't assign yourself just because you'd like to work on the issue, but only when you actually start.)
This helps avoid duplicate work.

If you start working on an issue but find that you won't be able to finish, please un-assign yourself so other people know the issue is available.
If you assign yourself but aren't making progress, we may assign the issue to someone else.

If you're working on issue 123, please put "Fixes #123" (without quotes) in the commit message below everything else and separated by a blank line.
For example, if issue 123 is a feature request to add foobar, the commit message might look like:

```text
Add foobar

Some longer description goes here, if you
want to describe your change in detail.

Fixes #123
```

This will [close the bug once your pull request is merged](https://help.github.com/articles/closing-issues-using-keywords/).

If you're a first-time contributor, try looking for an issue with the label "good first issue", which should be easier for someone unfamiliar with the codebase to work on.
