%% nirs_blocos_helper.m
% Funções para tratar corretamente um NIRS.mat que veio de arquivos
% concatenados. Use depois de gerar o NIRS.mat pela pipeline normal.
%
% Duas coisas precisam ser feitas bloco a bloco:
%
%   1. DETRENDING. O DCT global ajusta cossenos suaves ao sinal inteiro. Na
%      emenda existe uma descontinuidade, que cossenos suaves não conseguem
%      representar — o ajuste oscila em torno dela (ringing tipo Gibbs) e essa
%      oscilação se espalha para longe da emenda. Detrendar bloco a bloco faz
%      cada DCT enxergar só um trecho contínuo.
%
%   2. REGRESSORES DE BLOCO. Cada bloco tem o próprio nível basal. Sem uma
%      coluna constante por bloco na matriz de desenho, essa diferença vira
%      variância não modelada: infla o ResSS, derruba os t. Pior, se as
%      condições estiverem distribuídas de forma desigual entre blocos, o
%      offset fica correlacionado com o regressor e o beta sai enviesado.
%
% USO:
%   blocks = ler_blocos('/caminho/merged_blocks.mat');
%   Y.hbo  = detrend_por_bloco(Y.hbo, blocks, P.fs, P.K.H.cutoff);
%   X      = [X, regressores_de_bloco(blocks, P.ns)];
% =========================================================================


function blocks = ler_blocos(sidecar_path)
%LER_BLOCOS Carrega os limites dos blocos gravados pelo concatenador.
%   Retorna struct com campos starts, ends, n, names (índices em base 1).

    if ~exist(sidecar_path, 'file')
        error('Sidecar não encontrado: %s', sidecar_path);
    end
    S = load(sidecar_path);

    blocks.starts = double(S.blockStarts(:))';
    blocks.ends   = double(S.blockEnds(:))';
    blocks.n      = numel(blocks.starts);

    if isfield(S, 'blockNames')
        blocks.names = S.blockNames;
    else
        blocks.names = arrayfun(@(i) sprintf('bloco%d', i), 1:blocks.n, ...
                                'UniformOutput', false);
    end

    if numel(blocks.ends) ~= blocks.n
        error('Sidecar inconsistente: %d starts, %d ends.', ...
              blocks.n, numel(blocks.ends));
    end

    fprintf('Blocos carregados: %d\n', blocks.n);
    for i = 1:blocks.n
        fprintf('  bloco %d: amostras %d a %d (%d amostras)\n', ...
                i, blocks.starts(i), blocks.ends(i), ...
                blocks.ends(i) - blocks.starts(i) + 1);
    end
end


function blocks = blocos_do_nirs(nirs_path)
%BLOCOS_DO_NIRS Lê os limites direto do .nirs concatenado.
%   O concatenador grava blockStarts/blockEnds dentro do próprio arquivo,
%   então o sidecar é opcional.

    S = load(nirs_path, '-mat', 'blockStarts', 'blockEnds');
    if ~isfield(S, 'blockStarts')
        error(['%s não tem metadados de bloco. Ou não veio do concatenador, ' ...
               'ou é um arquivo de gravação única.'], nirs_path);
    end
    blocks.starts = double(S.blockStarts(:))';
    blocks.ends   = double(S.blockEnds(:))';
    blocks.n      = numel(blocks.starts);
    blocks.names  = arrayfun(@(i) sprintf('bloco%d', i), 1:blocks.n, ...
                             'UniformOutput', false);
end


function Y = detrend_por_bloco(Y, blocks, fs, cutoff)
%DETREND_POR_BLOCO Aplica DCT high-pass separadamente em cada bloco.
%   Y      : [ns x nch]
%   blocks : struct de ler_blocos
%   fs     : Hz
%   cutoff : período de corte em segundos

    ns = size(Y, 1);
    if blocks.ends(end) ~= ns
        warning(['Último bloco termina em %d mas o sinal tem %d amostras. ' ...
                 'Confira se blocos e dados são do mesmo arquivo.'], ...
                blocks.ends(end), ns);
    end

    for b = 1:blocks.n
        i0 = blocks.starts(b);
        i1 = min(blocks.ends(b), ns);
        seg = Y(i0:i1, :);

        nseg = size(seg, 1);
        N = max(1, fix(2 * nseg / (cutoff * fs)));
        K = spm_dctmtx(nseg, N + 1);
        K = K(:, 2:end);

        Y(i0:i1, :) = seg - K * (K' * seg);
    end
end


function R = regressores_de_bloco(blocks, ns)
%REGRESSORES_DE_BLOCO Colunas constantes, uma por bloco.
%   Modela explicitamente o offset de cada bloco para que ele não vaze para
%   os betas das condições. Retorna [ns x (nblocos-1)].
%
%   O último bloco fica de fora porque a matriz de desenho já tem uma
%   constante global — incluir todos os blocos deixaria o desenho singular.

    R = zeros(ns, max(0, blocks.n - 1));
    for b = 1:(blocks.n - 1)
        i0 = blocks.starts(b);
        i1 = min(blocks.ends(b), ns);
        R(i0:i1, b) = 1;
    end
end


function checar_eventos_nos_blocos(onsets_amostras, blocks, fs, pre_s)
%CHECAR_EVENTOS_NOS_BLOCOS Avisa se alguma janela pré-onset cruza uma emenda.
%   Quando isso acontece, a baseline da época é calculada num bloco e a
%   resposta em outro — a subtração de baseline deixa de cancelar o offset e
%   a época inteira sai deslocada.
%
%   onsets_amostras : vetor de onsets em amostras (base 1)
%   pre_s           : janela pré-onset usada na análise de época, em segundos

    pre_n = round(pre_s * fs);
    problemas = 0;

    for k = 1:numel(onsets_amostras)
        o = onsets_amostras(k);
        b = find(blocks.starts <= o & blocks.ends >= o, 1);
        if isempty(b)
            fprintf('  onset %d (amostra %d): fora de qualquer bloco\n', k, o);
            problemas = problemas + 1;
            continue;
        end
        if (o - pre_n) < blocks.starts(b)
            margem = (o - blocks.starts(b)) / fs;
            fprintf(['  onset %d (amostra %d, bloco %d): só %.1f s de margem, ' ...
                     'mas a janela pré pede %.1f s\n'], k, o, b, margem, pre_s);
            problemas = problemas + 1;
        end
    end

    if problemas == 0
        fprintf('  Todos os onsets têm margem suficiente antes do início do bloco.\n');
    else
        fprintf(['\n  %d onset(s) com margem insuficiente. Para esses, reduza a ' ...
                 'janela pré ou exclua a época.\n'], problemas);
    end
end
