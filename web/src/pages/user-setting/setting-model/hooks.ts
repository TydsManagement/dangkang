import { useSetModalState, useShowDeleteConfirm } from '@/hooks/commonHooks';
import {
  IApiKeySavingParams,
  ISystemModelSettingSavingParams,
  useAddLlm,
  useDeleteLlm,
  useFetchLlmList,
  useSaveApiKey,
  useSaveTenantInfo,
  useSelectLlmOptionsByModelType,
} from '@/hooks/llmHooks';
import { useOneNamespaceEffectsLoading } from '@/hooks/storeHooks';
import {
  useFetchTenantInfo,
  useSelectTenantInfo,
} from '@/hooks/userSettingHook';
import { IAddLlmRequestBody } from '@/interfaces/request/llm';
import { useCallback, useEffect, useState } from 'react';

type SavingParamsState = Omit<IApiKeySavingParams, 'api_key'>;

export const useSubmitApiKey = () => {
  const [savingParams, setSavingParams] = useState<SavingParamsState>(
    {} as SavingParamsState,
  );
  const saveApiKey = useSaveApiKey();
  const {
    visible: apiKeyVisible,
    hideModal: hideApiKeyModal,
    showModal: showApiKeyModal,
  } = useSetModalState();

  const onApiKeySavingOk = useCallback(
    async (apiKey: string, baseUrl: string) => {
      const ret = await saveApiKey({
        ...savingParams,
        api_key: apiKey,
        base_url: baseUrl,
      });

      if (ret === 0) {
        hideApiKeyModal();
      }
    },
    [hideApiKeyModal, saveApiKey, savingParams],
  );

  const onShowApiKeyModal = useCallback(
    (savingParams: SavingParamsState) => {
      setSavingParams(savingParams);
      showApiKeyModal();
    },
    [showApiKeyModal, setSavingParams],
  );

  const loading = useOneNamespaceEffectsLoading('settingModel', [
    'set_api_key',
  ]);

  return {
    saveApiKeyLoading: loading,
    initialApiKey: '',
    llmFactory: savingParams.llm_factory,
    onApiKeySavingOk,
    apiKeyVisible,
    hideApiKeyModal,
    showApiKeyModal: onShowApiKeyModal,
  };
};

export const useSubmitSystemModelSetting = () => {
  const systemSetting = useSelectTenantInfo();
  const loading = useOneNamespaceEffectsLoading('settingModel', [
    'set_tenant_info',
  ]);
  const saveSystemModelSetting = useSaveTenantInfo();
  const {
    visible: systemSettingVisible,
    hideModal: hideSystemSettingModal,
    showModal: showSystemSettingModal,
  } = useSetModalState();

  const onSystemSettingSavingOk = useCallback(
    async (
      payload: Omit<ISystemModelSettingSavingParams, 'tenant_id' | 'name'>,
    ) => {
      const ret = await saveSystemModelSetting({
        tenant_id: systemSetting.tenant_id,
        name: systemSetting.name,
        ...payload,
      });

      if (ret === 0) {
        hideSystemSettingModal();
      }
    },
    [hideSystemSettingModal, saveSystemModelSetting, systemSetting],
  );

  return {
    saveSystemModelSettingLoading: loading,
    onSystemSettingSavingOk,
    systemSettingVisible,
    hideSystemSettingModal,
    showSystemSettingModal,
  };
};

export const useFetchSystemModelSettingOnMount = (visible: boolean) => {
  const systemSetting = useSelectTenantInfo();
  const allOptions = useSelectLlmOptionsByModelType();
  const fetchLlmList = useFetchLlmList();
  const fetchTenantInfo = useFetchTenantInfo();

  useEffect(() => {
    if (visible) {
      fetchLlmList();
      fetchTenantInfo();
    }
  }, [fetchLlmList, fetchTenantInfo, visible]);

  return { systemSetting, allOptions };
};

export const useSelectModelProvidersLoading = () => {
  const loading = useOneNamespaceEffectsLoading('settingModel', [
    'my_llm',
    'factories_list',
  ]);

  return loading;
};

export const useSubmitOllama = () => {
  const loading = useOneNamespaceEffectsLoading('settingModel', ['add_llm']);
  const [selectedLlmFactory, setSelectedLlmFactory] = useState<string>('');
  const addLlm = useAddLlm();
  const {
    visible: llmAddingVisible,
    hideModal: hideLlmAddingModal,
    showModal: showLlmAddingModal,
  } = useSetModalState();

  const onLlmAddingOk = useCallback(
    async (payload: IAddLlmRequestBody) => {
      const ret = await addLlm(payload);
      if (ret === 0) {
        hideLlmAddingModal();
      }
    },
    [hideLlmAddingModal, addLlm],
  );

  const handleShowLlmAddingModal = (llmFactory: string) => {
    setSelectedLlmFactory(llmFactory);
    showLlmAddingModal();
  };

  return {
    llmAddingLoading: loading,
    onLlmAddingOk,
    llmAddingVisible,
    hideLlmAddingModal,
    showLlmAddingModal: handleShowLlmAddingModal,
    selectedLlmFactory,
  };
};

export const useSubmitVolcEngine = () => {
  const loading = useOneNamespaceEffectsLoading('settingModel', ['add_llm']);
  const [selectedVolcFactory, setSelectedVolcFactory] = useState<string>('');
  const addLlm = useAddLlm();
  const {
    visible: volcAddingVisible,
    hideModal: hideVolcAddingModal,
    showModal: showVolcAddingModal,
  } = useSetModalState();

  const onVolcAddingOk = useCallback(
    async (payload: IAddLlmRequestBody) => {
      const ret = await addLlm(payload);
      if (ret === 0) {
        hideVolcAddingModal();
      }
    },
    [hideVolcAddingModal, addLlm],
  );

  const handleShowVolcAddingModal = (llmFactory: string) => {
    setSelectedVolcFactory(llmFactory);
    showVolcAddingModal();
  };

  return {
    volcAddingLoading: loading,
    onVolcAddingOk,
    volcAddingVisible,
    hideVolcAddingModal,
    showVolcAddingModal: handleShowVolcAddingModal,
    selectedVolcFactory,
  };
};

export const useHandleDeleteLlm = (llmFactory: string) => {
  const deleteLlm = useDeleteLlm();
  const showDeleteConfirm = useShowDeleteConfirm();

  const handleDeleteLlm = (name: string) => () => {
    showDeleteConfirm({
      onOk: async () => {
        deleteLlm({ llm_factory: llmFactory, llm_name: name });
      },
    });
  };

  return { handleDeleteLlm };
};
