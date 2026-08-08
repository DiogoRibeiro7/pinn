# Security Policy

## Supported versions

This project is pre-1.0 and maintained by a single author. Only the latest release
receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it through GitHub's private vulnerability reporting, on the
[Security tab](https://github.com/DiogoRibeiro7/pinn/security/advisories/new) of
this repository. If that is unavailable to you, email
**dfr@esmad.ipp.pt** with `[pinn security]` in the subject.

Please include enough detail to reproduce: the version or commit, the affected
entry point, and a minimal example. A description of the impact you believe it has
is helpful, though not required.

Expect an acknowledgement within **7 days**. This is a personal project rather than
a funded one, so please treat that as a good-faith target rather than a guarantee.
You will be credited in the advisory and the changelog unless you prefer otherwise.

## Threat model

Understanding what this library does and does not defend against will tell you
whether a given behaviour is a bug.

### Model files are executable input

`pinn` loads checkpoints and serialised models with PyTorch. As of 0.1.0 all such
calls use `torch.load(..., weights_only=True)`, which restricts deserialisation to
tensors and simple types. That is a meaningful hardening step, but the safest
assumption remains: **treat a checkpoint from an untrusted source the way you would
treat an executable from an untrusted source.**

Note that checkpoints written before 0.1.0 used a pickle protocol the weights-only
loader cannot read, and will not load on this release. That is intentional.

### The cache deserialises its own data only

`pinn.optimization.caching` uses `pickle` to persist cache entries. It is designed
to read back only what it wrote. Pointing its cache directory at data from another
source, or at a directory another user can write to, breaks that assumption and
allows arbitrary code execution. Keep cache directories private to the running user.

### The serving layer is unauthenticated

`pinn.deployment.server` and `pinn.deployment.grpc_server` provide **no
authentication, authorisation, or rate limiting**. They bind `0.0.0.0` by default
because a container must, configurable through the `PINN_HOST` environment
variable. Do not expose them directly to a untrusted network. Put them behind a
reverse proxy or gateway that terminates TLS and handles authentication.

### Out of scope

- Resource exhaustion from deliberately large inputs to the serving layer. There is
  no request size limiting; use a proxy.
- Numerical inaccuracy, non-convergence, or misleading results from a trained model.
  These are correctness issues — please file them as ordinary bugs.
- Vulnerabilities in PyTorch, NumPy, or other dependencies. Report those upstream,
  though do tell us if `pinn` needs a version bump or a workaround.
