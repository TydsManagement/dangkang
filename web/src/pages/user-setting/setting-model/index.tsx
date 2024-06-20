import { ReactComponent as MoreModelIcon } from '@/assets/svg/more-model.svg';
import SvgIcon from '@/components/svg-icon';
import { useSetModalState, useTranslate } from '@/hooks/commonHooks';
import {
  LlmItem,
  useFetchLlmFactoryListOnMount,
  useFetchMyLlmListOnMount,
} from '@/hooks/llmHooks';
import {
  CloseCircleOutlined,
  SettingOutlined,
  UserOutlined,
} from '@ant-design/icons';
import {
  Avatar,
  Button,
  Card,
  Col,
  Collapse,
  CollapseProps,
  Divider,
  Flex,
  List,
  Row,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from 'antd';
import { useCallback } from 'react';
import SettingTitle from '../components/setting-title';
import { isLocalLlmFactory } from '../utils';
import ApiKeyModal from './api-key-modal';
import {
  useHandleDeleteLlm,
  useSelectModelProvidersLoading,
  useSubmitApiKey,
  useSubmitOllama,
  useSubmitSystemModelSetting,
  useSubmitVolcEngine,
} from './hooks';
import styles from './index.less';
import OllamaModal from './ollama-modal';
import SystemModelSettingModal from './system-model-setting-modal';
import VolcEngineModal from './volcengine-model';

const IconMap = {
  'Tongyi-Qianwen': 'tongyi',
  Moonshot: 'moonshot',
  OpenAI: 'openai',
  'ZHIPU-AI': 'zhipu',
  文心一言: 'wenxin',
  Ollama: 'ollama',
  Xinference: 'xinference',
  DeepSeek: 'deepseek',
  VolcEngine: 'volc_engine',
  BaiChuan: 'baichuan',
  Jina: 'jina',
};

const LlmIcon = ({ name }: { name: string }) => {
  const icon = IconMap[name as keyof typeof IconMap];

  return icon ? (
    <SvgIcon name={`llm/${icon}`} width={48} height={48}></SvgIcon>
  ) : (
    <Avatar shape="square" size="large" icon={<UserOutlined />} />
  );
};

const { Text } = Typography;
interface IModelCardProps {
  item: LlmItem;
  clickApiKey: (llmFactory: string) => void;
}

const ModelCard = ({ item, clickApiKey }: IModelCardProps) => {
  const { visible, switchVisible } = useSetModalState();
  const { t } = useTranslate('setting');
  const { handleDeleteLlm } = useHandleDeleteLlm(item.name);

  const handleApiKeyClick = () => {
    clickApiKey(item.name);
  };

  const handleShowMoreClick = () => {
    switchVisible();
  };

  return (
    <List.Item>
      <Card className={styles.addedCard}>
        <Row align={'middle'}>
          <Col span={12}>
            <Flex gap={'middle'} align="center">
              <LlmIcon name={item.name} />
              <Flex vertical gap={'small'}>
                <b>{item.name}</b>
                <Text>{item.tags}</Text>
              </Flex>
            </Flex>
          </Col>
          <Col span={12} className={styles.factoryOperationWrapper}>
            <Space size={'middle'}>
              <Button onClick={handleApiKeyClick}>
                {isLocalLlmFactory(item.name) || item.name === 'VolcEngine'
                  ? t('addTheModel')
                  : 'API-Key'}
                <SettingOutlined />
              </Button>
              <Button onClick={handleShowMoreClick}>
                <Flex gap={'small'}>
                  {t('showMoreModels')}
                  <MoreModelIcon />
                </Flex>
              </Button>
            </Space>
          </Col>
        </Row>
        {visible && (
          <List
            size="small"
            dataSource={item.llm}
            className={styles.llmList}
            renderItem={(item) => (
              <List.Item>
                <Space>
                  {item.name} <Tag color="#b8b8b8">{item.type}</Tag>
                  <Tooltip title={t('delete', { keyPrefix: 'common' })}>
                    <Button type={'text'} onClick={handleDeleteLlm(item.name)}>
                      <CloseCircleOutlined style={{ color: '#D92D20' }} />
                    </Button>
                  </Tooltip>
                </Space>
              </List.Item>
            )}
          />
        )}
      </Card>
    </List.Item>
  );
};

const UserSettingModel = () => {
  const factoryList = useFetchLlmFactoryListOnMount();
  const llmList = useFetchMyLlmListOnMount();
  const loading = useSelectModelProvidersLoading();
  const {
    saveApiKeyLoading,
    initialApiKey,
    llmFactory,
    onApiKeySavingOk,
    apiKeyVisible,
    hideApiKeyModal,
    showApiKeyModal,
  } = useSubmitApiKey();
  const {
    saveSystemModelSettingLoading,
    onSystemSettingSavingOk,
    systemSettingVisible,
    hideSystemSettingModal,
    showSystemSettingModal,
  } = useSubmitSystemModelSetting();
  const { t } = useTranslate('setting');
  const {
    llmAddingVisible,
    hideLlmAddingModal,
    showLlmAddingModal,
    onLlmAddingOk,
    llmAddingLoading,
    selectedLlmFactory,
  } = useSubmitOllama();

  const {
    volcAddingVisible,
    hideVolcAddingModal,
    showVolcAddingModal,
    onVolcAddingOk,
    volcAddingLoading,
    selectedVolcFactory,
  } = useSubmitVolcEngine();

  const handleApiKeyClick = useCallback(
    (llmFactory: string) => {
      if (isLocalLlmFactory(llmFactory)) {
        showLlmAddingModal(llmFactory);
      } else if (llmFactory === 'VolcEngine') {
        showVolcAddingModal('VolcEngine');
      } else {
        showApiKeyModal({ llm_factory: llmFactory });
      }
    },
    [showApiKeyModal, showLlmAddingModal, showVolcAddingModal],
  );

  const handleAddModel = (llmFactory: string) => () => {
    if (isLocalLlmFactory(llmFactory)) {
      showLlmAddingModal(llmFactory);
    } else if (llmFactory === 'VolcEngine') {
      showVolcAddingModal('VolcEngine');
    } else {
      handleApiKeyClick(llmFactory);
    }
  };

  const items: CollapseProps['items'] = [
    {
      key: '1',
      label: t('addedModels'),
      children: (
        <List
          grid={{ gutter: 16, column: 1 }}
          dataSource={llmList}
          renderItem={(item) => (
            <ModelCard item={item} clickApiKey={handleApiKeyClick}></ModelCard>
          )}
        />
      ),
    },
    {
      key: '2',
      label: t('modelsToBeAdded'),
      children: (
        <List
          grid={{
            gutter: 24,
            xs: 1,
            sm: 2,
            md: 3,
            lg: 4,
            xl: 4,
            xxl: 8,
          }}
          dataSource={factoryList}
          renderItem={(item) => (
            <List.Item>
              <Card className={styles.toBeAddedCard}>
                <Flex vertical gap={'large'}>
                  <LlmIcon name={item.name} />
                  <Flex vertical gap={'middle'}>
                    <b>{item.name}</b>
                    <Text>{item.tags}</Text>
                  </Flex>
                </Flex>
                <Divider></Divider>
                <Button type="link" onClick={handleAddModel(item.name)}>
                  {t('addTheModel')}
                </Button>
              </Card>
            </List.Item>
          )}
        />
      ),
    },
  ];

  return (
    <section id="xx" className={styles.modelWrapper}>
      <Spin spinning={loading}>
        <section className={styles.modelContainer}>
          <SettingTitle
            title={t('model')}
            description={t('modelDescription')}
            showRightButton
            clickButton={showSystemSettingModal}
          ></SettingTitle>
          <Divider></Divider>
          <Collapse defaultActiveKey={['1', '2']} ghost items={items} />
        </section>
      </Spin>
      <ApiKeyModal
        visible={apiKeyVisible}
        hideModal={hideApiKeyModal}
        loading={saveApiKeyLoading}
        initialValue={initialApiKey}
        onOk={onApiKeySavingOk}
        llmFactory={llmFactory}
      ></ApiKeyModal>
      <SystemModelSettingModal
        visible={systemSettingVisible}
        onOk={onSystemSettingSavingOk}
        hideModal={hideSystemSettingModal}
        loading={saveSystemModelSettingLoading}
      ></SystemModelSettingModal>
      <OllamaModal
        visible={llmAddingVisible}
        hideModal={hideLlmAddingModal}
        onOk={onLlmAddingOk}
        loading={llmAddingLoading}
        llmFactory={selectedLlmFactory}
      ></OllamaModal>
      <VolcEngineModal
        visible={volcAddingVisible}
        hideModal={hideVolcAddingModal}
        onOk={onVolcAddingOk}
        loading={volcAddingLoading}
        llmFactory={selectedVolcFactory}
      ></VolcEngineModal>
    </section>
  );
};

export default UserSettingModel;
