ARG BASE_IMAGE=lbx-tasks-base
ARG BASE_TAG=runtime-ml-core-py313-local
ARG PROBLEM_DIR=problems/openfoam-postproc-ddf10395
FROM ${BASE_IMAGE}:${BASE_TAG}

ARG PROBLEM_DIR=problems/openfoam-postproc-ddf10395
WORKDIR /workdir

RUN mkdir -p /mcp_server/data /workdir /tmp/output && chown -R 1000:1000 /workdir /tmp/output
COPY grader/ /mcp_server/grading/
RUN env -u UV_SYSTEM_PYTHON uv pip install --python /mcp_server/.venv/bin/python --no-cache -e /mcp_server/grading
RUN if [ ! -f /etc/solver-envs.d/openfoam.sh ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends openfoam \
      && rm -rf /var/lib/apt/lists/* \
      && mkdir -p /etc/solver-envs.d \
      && printf '%s\n' \
        'if [ -f /usr/share/openfoam/etc/bashrc ]; then' \
        '  source /usr/share/openfoam/etc/bashrc' \
        'elif [ -f /opt/openfoam*/etc/bashrc ]; then' \
        '  source /opt/openfoam*/etc/bashrc' \
        'fi' \
        > /etc/solver-envs.d/openfoam.sh; \
    fi
COPY ${PROBLEM_DIR}/data/ /data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/data/ /mcp_server/data/
COPY --chmod=0700 ${PROBLEM_DIR}/scorer/ /mcp_server/grader/
RUN chmod 0755 /mcp_server && chmod -R 0700 /mcp_server/data /mcp_server/grader

COPY ${PROBLEM_DIR}/task.toml ${PROBLEM_DIR}/instruction.md /task/
