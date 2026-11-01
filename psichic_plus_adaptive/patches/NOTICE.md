# License notices

AdaptiveFlow-VS is an external dependency from [QuantumAI4Bio/AdaptiveFlow-VS](https://github.com/QuantumAI4Bio/AdaptiveFlow-VS).
The supported revision is `df74e005e456e94654fa93a40d748de399450b81`.
Its repository includes the GNU General Public License version 2.
Its modified worker and Bash template have notices allowing GPL version 2 or later.
The [upstream license text](LICENSE.AdaptiveFlow) is retained here.

`af-vs-df74e005.patch` modifies those upstream files and is provided under GPL-2.0-or-later.
The original files credit Christoph Gorgulla and Amazon.com, Inc. or its affiliates.
The patch adds the lazy `external_batch` entry point, handles unavailable queue-size queries on macOS, allows a local temporary directory, and propagates worker exit failures from the Bash template.
The launcher applies this checked patch to a local clone, rather than distributing a modified copy of upstream source.

PSICHIC+ and this repository's adapter code are subject to the repository's [non-commercial research license](../../LICENSE).
Model weights and the ESM backbone have their own applicable terms.
The integration Dockerfile copies no model weights and accepts a user-specified base image.
The integration does not grant rights to distribute weights or a combined image.

These are separate licenses.
Keeping the dependency external, using processes, or applying a patch does not establish license compatibility.
This integration makes no assertion of compatibility between GPL terms and the PSICHIC+ license.
Review the applicable terms and obtain any required permissions before distribution or use beyond those terms.
